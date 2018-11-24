from run_common import run

run(dict(
    data_dir='../huawei_data/transformed',
    test_split=0.2,
    data_subset=0.01,
    workers=4,
    save_dir='',
    epochs=10,
    train_batch_size=256,
    test_batch_size=256,
    learning_rate=0.005,
    loss='MSELoss',
    model='SimpleCNN',
    optim='Adam',
    gpu_num=0,
    cnn_in_channels=3,
    cnn_hidden_channels=32,
    cnn_hidden_layers=7,
))
