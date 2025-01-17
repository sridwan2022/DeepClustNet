import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans
from Model import Autoencoder, generator
import metrics
import time
import math
import numpy as np
import os
import csv

class DeepClustNet(nn.Module):
    def __init__(self, dims, n_clusters):
        super(DeepClustNet, self).__init__()
        self.dims = dims
        self.n_clusters = n_clusters
        self.centers = []
        self.y_pred = []

        self.autoencoder, self.encoder = Autoencoder(dims=dims)
        self.model = self.encoder
        self.pretrained = False

    def pretrain(self, x, y=None, optimizer=torch.optim.Adam, epochs=200, batch_size=256, save_dir='results/temp',
                 da_s1=False, verbose=1, use_multiprocessing=True):
        print('Autoencoder training......')
        self.autoencoder.compile(optimizer=optimizer(self.autoencoder.parameters()), loss=nn.MSELoss())

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        csv_logger = csv.writer(open(os.path.join(save_dir, 'pretrain_log.csv'), 'w'))

        # begin pretraining
        t0 = time.time()
        if not da_s1:
            dataloader = DataLoader(x, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                for batch in dataloader:
                    self.autoencoder.zero_grad()
                    outputs = self.autoencoder(batch)
                    loss = nn.MSELoss()(outputs, batch)
                    loss.backward()
                    optimizer.step()
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
        else:
            print('-=*' * 20)
            print('Using augmentation for pretraining')
            print('-=*' * 20)

            dataloader = DataLoader(x, batch_size=batch_size, steps_per_epoch=math.ceil(x.shape[0] / batch_size), epochs=epochs)
            for epoch in range(epochs):
                for batch in dataloader:
                    self.autoencoder.zero_grad()
                    outputs = self.autoencoder(batch)
                    loss = nn.MSELoss()(outputs, batch)
                    loss.backward()
                    optimizer.step()
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))

        print('Autoencoder training time: ', time.time() - t0)
        torch.save(self.autoencoder.state_dict(), os.path.join(save_dir, 'ae_weights.pth'))
        print('Autoencoder training weights are saved to %s/ae_weights.pth' % save_dir)
        self.pretrained = True

    def basic_clustering(self, x):
        """ Initialize a clustering result, i.e., labels and cluster centers.
        :param x: input data, shape=[n_samples, n_features]
        :return: labels and centers
        """
        print("Using k-means for initialization by default.")
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        print(x.shape)
        y_pred = kmeans.fit_predict(X=x)
        centers = kmeans.cluster_centers_.astype(np.float32)
        return y_pred, centers

    def self_paced_learning(data_loader, model, optimizer, criterion, epoch):
        alpha = 0.2  # Initial self-paced learning weight
        max_alpha = 1.0  # Maximum self-paced learning weight
        for i, (inputs, _) in enumerate(data_loader):
            inputs = inputs.view(inputs.size(0), -1)
            inputs = Variable(inputs.cuda())
            optimizer.zero_grad()

            # Compute reconstruction loss and clustering loss
            reconstructed, clustering, weighted_encoded = model(inputs)
            reconstruction_loss = criterion(reconstructed, inputs)
            clustering_loss = criterion(clustering, targets)  # Define your own clustering loss function

            # Compute the overall loss with the self-paced learning weight
            loss = reconstruction_loss + alpha * clustering_loss

            # Update the model parameters
            loss.backward()
            optimizer.step()

            # Update self-paced learning weight
            alpha = min(alpha + 0.01, max_alpha)  # Increase the alpha gradually during training epochs

    def update_labels(self, x, centers):
        """ Update cluster labels.
        :param x: input data, shape=(n_samples, n_features)
        :param centers: cluster centers, shape=(n_cluster, n_features)
        :return: (labels, loss): labels indicate each sample belongs to which cluster. labels[i]=j means sample i
                 belongs to cluster j; loss, the average distance between samples and their responding centers
        """
        x_norm = np.reshape(np.sum(np.square(x), 1), [-1, 1])  # column vector
        center_norm = np.reshape(np.sum(np.square(centers), 1), [1, -1])  # row vector
        dists = x_norm - 2 * np.matmul(x, np.transpose(centers)) + center_norm  # |x-y|^2 = |x|^2 -2*x*y^T + |y|^2
        labels = np.argmin(dists, 1)
        losses = np.min(dists, 1)
        return labels, losses

    def compute_sample_weight(self, losses, t, T):
        lam = np.mean(losses) + t*np.std(losses) / T
        return np.where(losses < lam, 1., 0.)

    def load_weights(self, weights):  # load weights of DEC model
        self.model.load_state_dict(torch.load(weights))

    def predict(self, x, batch_size=256):
        dataloader = DataLoader(x, batch_size=batch_size)
        self.model.eval()
        with torch.no_grad():
            for batch in dataloader:
                outputs = self.model(batch)
        return outputs

    def predict_labels(self, x):  # predict cluster labels using the output of clustering layer
        return self.basic_clustering(self.predict(x))[0]

    def get_labels(self):
        return self.y_pred

    def compile(self, optimizer=torch.optim.SGD, loss=nn.MSELoss()):
        self.model.compile(optimizer=optimizer(self.model.parameters()), loss=loss)

    def fit(self, x, y=None, batch_size=256, epochs=100,
            ae_weights=None, save_dir='result/temp', tol=0.001,
            use_sp=True, da_s2=False, use_multiprocessing=True):

        # prepare folder for saving results
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # pretraining
        t0 = time.time()
        if ae_weights is None and not self.pretrained:
            print('Pretraining AE...')
            self.pretrain(x, save_dir=save_dir)
            print('Pretraining time: %.1fs' % (time.time() - t0))
        elif ae_weights is not None:
            self.autoencoder.load_state_dict(torch.load(ae_weights))
            print('Pretrained AE weights are loaded successfully!')

        # initialization
        t1 = time.time()
        self.y_pred, self.centers = self.basic_clustering(self.predict(x))
        t2 = time.time()
        print('Time for initialization: %.1fs' % (t2 - t1))

        # logging file
        logfile = open(save_dir + '/log.csv', 'w')
        logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'acc', 'nmi', 'Ln', 'Lc'])
        logwriter.writeheader()

        net_loss = 0
        clustering_loss = 0
        time_train = 0
        sample_weight = np.ones(shape=x.shape[0])
        sample_weight[self.y_pred == -1] = 0  # do not use the noisy examples
        y_pred_last = np.copy(self.y_pred)
        result = None
        for epoch in range(epochs+1):
            """ Log and check stopping criterion """
            if y is not None:
                acc = np.round(metrics.acc(y, self.y_pred), 5)
                nmi = np.round(metrics.nmi(y, self.y_pred), 5)
                print('Epoch-%d: ACC=%.4f, NMI=%.4f, Ln=%.4f, Lc=%.4f; time=%.1f' %
                      (epoch, acc, nmi, net_loss, clustering_loss, time_train))
                logwriter.writerow(dict(epoch=epoch, acc=acc, nmi=nmi, Ln=net_loss, Lc=clustering_loss))
                logfile.flush()

                # record the initial result
                if epoch == 0:
                    print('DeepClustNet model saved to \'%s/model_init.pth\'' % save_dir)
                    torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_init.pth'))

                # check stop criterion
                delta_y = np.sum(self.y_pred != y_pred_last).astype(np.float32) / self.y_pred.shape[0]
                y_pred_last = np.copy(self.y_pred)
                if (epoch > 0 and delta_y < tol) or epoch >= epochs:
                    result = np.asarray([acc, nmi])
                    print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, tol))
                    print('DeepClustNet model saved to \'%s/model_final.pth\'' % save_dir)
                    print('-' * 30 + ' END: time=%.1fs ' % (time.time()-t0) + '-' * 30)
                    torch.save(self.model.state_dict(), os.path.join(save_dir, 'model_final.pth'))
                    logfile.close()
                    break

            """ Step 1: train the network """
            t0_epoch = time.time()
            if da_s2:  # use data augmentation
                dataloader = DataLoader(x, self.centers[self.y_pred], sample_weight, batch_size,
                                        steps_per_epoch=math.ceil(x.shape[0] / batch_size), epochs=5 if np.any(self.y_pred == -1) and epoch==0 else 1)
            else:
                dataloader = DataLoader(x, batch_size=batch_size, shuffle=True)
            for batch in dataloader:
                self.model.zero_grad()
                outputs = self.model(batch)
                loss = nn.MSELoss()(outputs, self.centers[self.y_pred])
                loss.backward()
                optimizer.step()
            net_loss = loss.item()

            """ Step 2: update labels """
            self.y_pred, losses = self.update_labels(self.predict(x), self.centers)
            clustering_loss = np.mean(losses)

            """ Step 3: Compute sample weights """
            sample_weight = self.compute_sample_weight(losses, epoch, epochs) if use_sp else None

            time_train = time() - t0_epoch

        return result