import warnings

class DefaultConfig(object):

    env = 'default'

    #dataset_train_path = './data/train/'
    #dataset_test_path = './data/test/'

    train_data_root = './data/train/'
    test_data_root = './data/test/'
    load_model_path = None
    load_model = None
    pretrained_model_path = None
    customized_model = 'CustomizedNet'

    model = 'AlexNet'

    use_gpu = False
    num_workers = 4
    print_freq = 20

    #hyper-parameter
    batch_size = 64
    lr = 0.001
    epoches = 1
    lr_decay = 0.95
    weight_decay = 1e-4

    # enable utils
    profiling = False
    # ddpg config
    gamma = 0.99
    tau = 0.001
    noise_scale = 0.5
    final_noise_scale = 0.5
    warmup = 20
    exploration_end = 100
    seed = 58
    sample_size = 128
    num_episodes = 150
    hidden_size = 128
    updates_per_step = 3
    replay_size = 100000
def parse(self, kwargs):

    for k, v in kwargs.items():
        if(not hasattr(self, k)):
            warnings.warn('Warning: opt has not attribute %s' % k)
        setattr(self, k, v)

    print('\n## user config --------------------------------------------\n')
    for k, v in self.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(self, k))
    print('\n-----------------------------------------------------------\n')

DefaultConfig.parse = parse
opt = DefaultConfig()
