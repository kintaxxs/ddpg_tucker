import torch
import fire
from time import time
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config import opt
import torchvision
from utils.decompositions import tucker_decomposition_conv_layer, estimate_ranks, tucker_decomposition_fc_layer
from utils.decompositions import tucker_decomposition_conv_layer_without_rank
import tensorly as tl
import torchvision.transforms as transforms
import os
import torch.backends.cudnn as cudnn
from utils.utils import progress_bar
import copy
from ddpg.ddpg import DDPG
from ddpg.ounoise import OUNoise
from ddpg.replay_memory import ReplayMemory, Transition
import numpy as np
from logger import Logger
logger = Logger('/train-log/red/logs/')
def ddpg_decomposition(**kwargs):
    opt.parse(kwargs)
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    tl.set_backend('pytorch')
    # Model, Load checkpoint
    print('==> Load checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if opt.load_model:
        checkpoint = torch.load('checkpoint/'+opt.load_model)
    else:
        import sys 
        print('set the load_model_path')
        sys.exit(1)
    model = checkpoint['net']
    decomp_model = copy.deepcopy(model)
    #best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    print(model)
    model.cuda()
    decomp_model.cuda()
    N_feature = len(model.features._modules.keys())
    N_classifier = len(model.classifier._modules.keys())
    ft = model.features._modules
    states = []
    origin_flops = [7077888, 10616832, 14155776, 9437184]
    j=0
    for i, key in enumerate(model.features._modules.keys()):
        if i == 0:
            continue
        if isinstance(ft[key], torch.nn.modules.conv.Conv2d):
            states.append([int(key), 
                getattr(ft[key],'in_channels'),
                getattr(ft[key],'out_channels'),
                #getattr(ft[key],'kernel_size')[0],
                #getattr(ft[key],'stride')[0],
                #getattr(ft[key],'padding')[0],
                np.log10(origin_flops[j])
            ])
            j+=1
    print(states)
    action_space = torch.Tensor(2,)
    agent = DDPG(opt.gamma, opt.tau, opt.hidden_size, len(states[0]), action_space)
    memory = ReplayMemory(opt.replay_size)
    ounoise = OUNoise(action_space.shape[0])
    rewards = []
    best_reward = -10000
    best_model = None
    for i_episode in range(opt.num_episodes):
        state = torch.Tensor([states[0]])
        decomp_model = copy.deepcopy(model)
        decomp_model.cuda()
        ounoise.scale = (opt.noise_scale - opt.final_noise_scale) * max(0, opt.exploration_end -
                                    i_episode) / opt.exploration_end + opt.final_noise_scale
        ounoise.reset()
        episode_reward = 0
        total_flops = 0
        for t in range(len(states)):
            key = str(states[t][0])
            rank = None
            #Phase action seletion
            #if i_episode == 0:
                #rank = estimate_ranks(model.features._modules[key])
                #action = torch.Tensor([float(rank[0])/ft[key].in_channels, float(rank[1])/ft[key].out_channels])
                #action = action.view(1,2)
            if i_episode < opt.warmup:
                action = torch.rand(1,2)
                rank = np.ceil([action.numpy()[0][0]*ft[key].in_channels,action.numpy()[0][1]*ft[key].out_channels]).astype(np.int32).clip(1,)
            elif i_episode < opt.exploration_end:
                action = agent.select_action(state,ounoise)
                print(action.data.numpy())
                rank = np.ceil([action.data.numpy()[0][0]*ft[key].in_channels,
                        action.data.numpy()[0][1]*ft[key].out_channels]).astype(np.int).clip(1,)
            else:
                action = agent.select_action(state)
                rank = np.ceil([action.data.numpy()[0][0]*ft[key].in_channels,
                        action.data.numpy()[0][1]*ft[key].out_channels]).astype(np.int).clip(1,)
            #Do action
            decompose = tucker_decomposition_conv_layer_without_rank(model.features._modules[key],rank)
            decomp_model.features._modules[key] = decompose
            #fine_tune(decomp_model, trainloader,testloader, opt.load_model, opt.use_gpu, False)
            print('In: {}, Out: {}'.format(ft[key].in_channels, ft[key].out_channels))
            accuracy = fine_tune_test(decomp_model, testloader, opt.use_gpu, 0, False)
            print('acc: ', accuracy)
            ratio = (rank[0] * rank[1]) / (ft[key].in_channels * ft[key].out_channels)
            print('Ratio: {}, FLOPs: {}, reduced: {}'.format(ratio,state.numpy()[0][-1], 
                state.numpy()[0][-1]+ np.log10(ratio)))
            reward = -(1.0- accuracy/100) * (state.numpy()[0][-1]+ np.log10(ratio))
            total_flops += np.power(10,state.numpy()[0][-1]) * ratio 
            next_state = torch.Tensor([states[(t+1) % len(states)]])
            done = True if (t+1) == len(states) else False
            mask = torch.Tensor([not done])
            memory.push(state, torch.Tensor(action), mask, next_state, torch.Tensor([reward]))
            state = next_state   
            episode_reward += reward
            if len(memory) > opt.sample_size * 3:
                print('Sample experience')
                for _ in range(opt.updates_per_step):
                    transitions = memory.sample(opt.sample_size)
                    batch = Transition(*zip(*transitions))

                    agent.update_parameters(batch)
            if done:
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    best_model = copy.deepcopy(decomp_model)
                episode_log = {
                    'acc': accuracy,
                    'flops': total_flops,
                    'episode_reward': episode_reward
                }
                for tag, value in episode_log.items():
                    logger.scalar_summary(tag, value, i_episode)
                model_state = {
                    'i_episode': i_episode,
                    'net': best_model,
                    'acc': accuracy,
                    'flops' : total_flops
                }
                if not os.path.isdir('/train-log/red/checkpoint'):
                    os.mkdir('/train-log/red/checkpoint')
                torch.save(model_state, '/train-log/red/checkpoint/'+'ddpg_best_noft_'+str(i_episode))
                break
        rewards.append(episode_reward)
        print("Episode: {}, noise: {}, reward: {}, total flops:{}, max reward: {}".format( \
            i_episode, ounoise.scale, rewards[-1], total_flops, best_reward))
def decomposition(**kwargs):
    opt.parse(kwargs)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    tl.set_backend('numpy')
    opt.parse(kwargs)

    # Model, Load checkpoint
    print('==> Load checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if opt.load_model:
        checkpoint = torch.load('checkpoint/'+opt.load_model)
    else:
        import sys 
        print('set the load_model_path')
        sys.exit(1)
    model = checkpoint['net']
    #best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
    print(model)

#    model.eval()
    model.cpu()
    N_feature = len(model.features._modules.keys())
    N_classifier = len(model.classifier._modules.keys())
    
    for i, key in enumerate(model.features._modules.keys()):
        print(i)
        if i == 0:
            continue
        if i >= N_feature - 2:
            break
        if isinstance(model.features._modules[key], torch.nn.modules.conv.Conv2d):
            print(i, 'decompose')
            conv_layer = model.features._modules[key]
            decomposed = tucker_decomposition_conv_layer(conv_layer)
            model.features._modules[key] = decomposed
    '''
    for i, key in enumerate(model.classifier._modules.keys()):
        print('i, N_classifier-2: ', i, N_classifier-2)
        if i >= N_classifier - 2:
            break
        if isinstance(model.classifier._modules[key], torch.nn.modules.linear.Linear):
            fc_layer = model.classifier._modules[key]
            decomposed = tucker_decomposition_fc_layer(fc_layer)
            model.classifier._modules[key] = decomposed
    '''
    acc = fine_tune_test(model, testloader, opt.use_gpu, 0, False)

    #torch.save(model, 'checkpoints/test_alexnet_model')
    print('final saving..')
    state = {
       'net': model,
       'acc': acc,
       'epoch': start_epoch,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/'+'decomposed_conv_'+opt.load_model)
    print(model)
    fine_tune(model, trainloader,testloader, opt.load_model, opt.use_gpu, False)
    
def print_model(**kwargs):
    opt.parse(kwargs)

    if opt.load_model_path:
            model = torch.load(opt.load_model_path)
            print(model['i_episode'])
            print(model['net'])
            print(model['acc'])
            print(model['flops'])
    else:
        import sys 
        print('set the load_model_path')
        sys.exit(1)
def ft(**kwargs):
    opt.parse(kwargs)
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    epoches = 20
    lr = 0.001
    use_gpu = True
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    if opt.load_model_path:
        ckpt = torch.load(opt.load_model_path)
        model = ckpt['net']
        if use_gpu:
            model.cuda()
            cudnn.benchmark = True
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    for epoch in range(epoches):

        train_loss = 0
        correct = 0
        total = 0

        print('\nEpoch: %d' % epoch)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = Variable(inputs), Variable(targets)
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
	        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        best_acc = fine_tune_test(model, testloader, True, best_acc, 0)
    model_state = {
        'i_episode': ckpt['i_episode'],
        'net': model,
        'acc': best_acc,
        'flops' : ckpt['flops']
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(model_state, './checkpoint/'+'vbmf_ft')
def fine_tune(model, trainloader, testloader, model_name, use_gpu, gradual_rank):

    model.train()

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    epoches = 5
    prev_loss = 0 
    lr = 0.001

    # Data
    if use_gpu:
        model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for epoch in range(epoches):

        train_loss = 0
        correct = 0
        total = 0

        print('\nEpoch: %d' % epoch)
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
	        #% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        best_acc = fine_tune_test(model, testloader, use_gpu, best_acc, 0)
        '''
        if(epoch == epoches-1):
	    # Save checkpoint.
            #acc = 100.*correct/total
            print('Final saving..')
            print('The best acc is ', best_acc)
            state = {
	        'net': model.module if use_gpu else model,
	        'acc': best_acc,
	    }

            if(not gradual_rank):
                file_prex = 'finetune'
            else:
                file_prex = 'gradual_rank_finetune'

            if not os.path.isdir(file_prex):
                os.mkdir(file_prex)
            torch.save(state, './'+file_prex+'/'+model_name)
        '''
def fine_tune_test(model, testloader, use_gpu, best_acc, gradual_rank):

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if use_gpu:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            if use_gpu:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        test_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*float(correct)/total, correct, total))
    model.train()
    acc = 100.*float(correct)/total
    if acc > best_acc:
        best_acc = acc
    return best_acc 

def help():

    print('''
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example
        python {0} train --env='env1025' --lr=0.01
        python {0} test --dataset='path/to/detaset/root/'
        python {0} help
    available args:'''.format(__file__))

    from inspect import getsource
    source = getsource(opt.__class__)
    print(source)

if __name__ == '__main__':
    fire.Fire()
