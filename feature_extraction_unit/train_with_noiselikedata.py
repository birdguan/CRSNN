from feature_extraction_unit.lane_extract import *
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    EPOCHNUM = 400

    kernels = [utils.DoGKernel(3, 3 / 9, 6 / 9),
               utils.DoGKernel(3, 6 / 9, 3 / 9),
               utils.DoGKernel(7, 7 / 9, 14 / 9),
               utils.DoGKernel(7, 14 / 9, 7 / 9),
               utils.DoGKernel(13, 13 / 9, 26 / 9),
               utils.DoGKernel(13, 26 / 9, 13 / 9)]
    filter = utils.Filter(kernels, padding=6, thresholds=50)
    s1c1 = TemporalFilter(filter)

    data_root = "data"
    noiseLikeData_train = ImageFolder("data", transform=transforms.Compose([transforms.ToTensor()]))
    noiseLikeData_loader = DataLoader(noiseLikeData_train, batch_size=200, shuffle=False)

    network = FeatureExtractionModel()
    if use_cuda:
        network.cuda()
    blank_image = np.zeros([32, 32])
    # Training The First Layer
    print("Training the first layer ...")
    cascade_tensor = torch.tensor(np.array([blank_image for _ in range(6)]))
    if os.path.isfile("layer1_model.net"):
        network.load_state_dict(torch.load("layer1_model.net"))
        print("pre-trained layer-1 model loaded")
    else:
        for epoch in range(2):
            iter = 0
            for data, targets in noiseLikeData_loader:
                data, cascade_tensor = stochastic_decay_gpu(cascade_tensor, data, stochastic_prob=0.2, decay_rate=0.1)
                iter += 1
                train_unsupervise(network, data, 1)
                if iter % 10 == 0:
                    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("==> %s [layer 1 epoch %d, iteration %d] Done" % (curr_time, epoch, iter))
        torch.save(network.state_dict(), "layer1_model.net")
    # Training The Second Layer
    print("Training the second layer ...")
    cascade_tensor = torch.tensor(np.array([blank_image for _ in range(6)]))
    if os.path.isfile("layer2_model.net"):
        network.load_state_dict(torch.load("layer2_model.net"))
        print("pre-trained layer-2 model loaded")
    else:
        for epoch in range(4):
            iter = 0
            for data, targets in noiseLikeData_loader:
                data, cascade_tensor = stochastic_decay_gpu(cascade_tensor, data, stochastic_prob=0.2, decay_rate=0.1)
                iter += 1
                train_unsupervise(network, data, 2)
                if iter % 10 == 0:
                    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("==> %s [layer 2 epoch %d, iteration %d] Done" % (curr_time, epoch, iter))
    torch.save(network.state_dict(), "layer2_model.net")

    # Training The third Layer
    print("Training the third layer ...")
    cascade_tensor = torch.tensor(np.array([blank_image for _ in range(6)]))
    if os.path.isfile("layer3_model.net"):
        network.load_state_dict(torch.load("layer3_model.net"))
        print("pre-trained layer-3 model loaded")
    else:
        for epoch in range(6):
            iter = 0
            for data, targets in noiseLikeData_loader:
                data, cascade_tensor = stochastic_decay_gpu(cascade_tensor, data, stochastic_prob=0.2, decay_rate=0.1)
                iter += 1
                train_unsupervise(network, data, 3)
                if iter % 10 == 0:
                    curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    print("==> %s [layer 3 epoch %d, iteration %d] Done" % (curr_time, epoch, iter))
    torch.save(network.state_dict(), "layer3_model.net")


