import random
import os

import numpy as np
import torch.optim as optim
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch.utils.data
from svaemodels.model import SVaeGRU
import time
from dataset.data_loader import make_dataloader
from dataset.increment import increment, sample_extend0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

def svae_gru(ori_data, D, data_name, n_epoch, save_path, state='train'):
    lr = 5e-4
    # n_epoch = 200
    manual_seed = random.randint(1, 10000)
    num_works = 8
    shuffle_train = True
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    batch_size = 16
    window_size = 1
    rnn_size = 128
    latent_size = 64
    hidden_size = 50
    stack_n = 2
    (num, T) = ori_data.shape
    processed_dir = 'data/processed'
    best_model_output_dir = 'output/'+str(data_name)+str(num)+'_'+str(T)+'_'+str(n_epoch)+'/'
    normalization = 'min-max'
    ori, generate = [], []

    # source
    train_loader, valid_loader, test_loader = make_dataloader(ori_data,
                                                              batch_size=batch_size,
                                                              num_workers=num_works,
                                                              shuffle=shuffle_train,
                                                              processed_dir='data/processed',
                                                              window_size=window_size,
                                                              normalization=normalization,
                                                              data_name=data_name
                                                              )
    dataloaders = {"train": train_loader, "valid": valid_loader, "test": test_loader}
    # load model
    model = SVaeGRU(input_size=window_size,
                         rnn_size=rnn_size,
                         latent_size=latent_size,
                         hidden_size=hidden_size,
                         stack_layer=stack_n,
                     )

    optimizer = optim.RMSprop(model.parameters(), lr=lr) # RMSprop in paper # Adam
    loss_fn = torch.nn.L1Loss() # MAE loss in paper # MSE
    model = model.to(DEVICE)
    loss_fn = loss_fn.to(DEVICE)

    for p in model.parameters():
        p.requires_grad = True

    if state == 'train':
        # training
        total_anneal_steps = 200000
        update_count = 0.0
        anneal_cap = 0.2
        since = time.time()
        best_loss = 10e5
        save_test = False
        for epoch in range(n_epoch):
            print(f'Epoch {epoch}/{n_epoch - 1}')
            print('-' * 10)
            ori_data_list, gen_data = [], []

            for phase in ["train", "test"]: #"valid", "test"
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()

                running_loss = 0.0
                batch_cnt = 0
                for input_batch in tqdm(dataloaders[phase]):
                    input_batch = input_batch.float().to(DEVICE)
                    optimizer.zero_grad()
                    gen, kl_loss = model(input_batch)
                    # gen, kl_loss, x_list = model(input_batch)
                    mae_loss = loss_fn(gen, input_batch)
                    # mse_loss = 0
                    '''for i in range(len(x_list)):
                        mae_loss += loss_fn(x_list[i], input_batch)'''

                    if epoch == n_epoch-1:
                        for i in range(len(input_batch)):
                            ori_data_list.append(np.array(input_batch[i].flatten().cpu().detach().tolist()).T)
                            gen_data.append(np.array(gen[i].flatten().cpu().detach().tolist()).T)
                            '''if data_name != 'Fatigue':
                                ori_data_list.append(np.array([0] + input_batch[i].flatten().cpu().detach().tolist()).T)
                                gen_data.append(np.array([0] + gen[i].flatten().cpu().detach().tolist()).T)
                            else:
                                ori_data_list.append(np.array(input_batch[i].flatten().cpu().detach().tolist()).T)
                                gen_data.append(np.array(gen[i].flatten().cpu().detach().tolist()).T)'''

                    if total_anneal_steps > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = anneal_cap
                    # total_loss = anneal * mse_loss + kl_loss
                    # total_loss = mse_loss + kl_loss
                    total_loss = mae_loss

                    if phase == "train":
                        # update_count += 1.0
                        total_loss.backward()
                        optimizer.step()
                    running_loss += total_loss.item() * input_batch.size(0)
                    batch_cnt += 1
                epoch_loss = running_loss / (batch_cnt * batch_size)
                print(f'{phase} Loss: {epoch_loss:.4f}')
                if phase == 'train' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    if not os.path.exists(best_model_output_dir):
                        os.makedirs(best_model_output_dir)
                    torch.save(model.state_dict(), os.path.join(best_model_output_dir, "epoch_{}_loss_{}.pt".format(epoch, epoch_loss)))
                    save_path = os.path.join(best_model_output_dir, "epoch_{}_loss_{}.pt".format(epoch, epoch_loss))
                    print('save modle to {}'.format(os.path.join(best_model_output_dir, "epoch_{}_loss_{}.pt".format(epoch, epoch_loss))))
                    save_test = True

                if phase == 'test' and save_test:
                    # loss_picture(input_batch[0].cpu().detach().numpy(), gen[0].cpu().detach().numpy(), epoch_loss, "epoch {} image".format(epoch), data_name)
                    save_test = False

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')
        print('Start evaluating')

        #print(ori_data)
        #print(gen_data)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        x_ticks = np.arange(0, input_batch.shape[1] + 1, 1)
        x = np.linspace(0, input_batch.shape[1], input_batch.shape[1] + 1)
        '''if data_name != 'Fatigue':
            x_ticks = np.arange(0, input_batch.shape[1] + 1, 1)
            x = np.linspace(0, input_batch.shape[1], input_batch.shape[1] + 1)
        else:
            x_ticks = np.arange(0, input_batch.shape[1], 1)
            x = np.linspace(0, input_batch.shape[1]-1, input_batch.shape[1])'''

        #constant = np.zeros((len(ori_data_list), 1))[:, 0]
        constant = ori_data[:len(ori_data_list), 0]
        ori = sample_extend0(np.array(ori_data_list), constant)
        generate = sample_extend0(np.array(gen_data), constant)

        od = ax.plot(x, ori.T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        sd = ax.plot(x, generate.T, 'm^--', label='SVAE-GRU')
        plt.setp(sd[1:], label="_")

        '''od = ax.plot(x, np.array(ori_data_list).T, 'bo--', label='Original')
        plt.setp(od[1:], label="_")
        sd = ax.plot(x, np.array(gen_data).T, 'm^--', label='SVAE-GRU')
        plt.setp(sd[1:], label="_")'''

        plt.xlabel("Time")
        plt.xticks(x_ticks)
        ax.legend()
        #plt.show()
        plt.savefig(best_model_output_dir + 'generation.png')
        plt.cla()
        plt.close('all')

    # Prediction
    loaded_model = SVaeGRU(input_size=window_size,
                               rnn_size=rnn_size,
                               latent_size=latent_size,
                               hidden_size=hidden_size,
                               stack_layer=stack_n,
                               ).to(DEVICE)
    loss_fn = torch.nn.MSELoss()  # MAE loss in paper
    loss_fn = loss_fn.to(DEVICE)
    loaded_model.load_state_dict(torch.load(save_path))
    loaded_model.eval()
    cnt = 1
    test_data, pre_data = [], []
    data_list, out_list = [], []
    numb = 0

    with torch.no_grad():
        for test_batch in tqdm(dataloaders['test']):
            test_batch = test_batch.float().to(DEVICE)
            optimizer.zero_grad()
            pre, _ = loaded_model(test_batch)
            mae_loss = loss_fn(test_batch, pre)
            in_list = [e[0] for e in test_batch.tolist()[0]]
            gen_list = [e[0] for e in pre.tolist()[0]]
            in_list, gen_list = evaluate_loss_picture(in_list, gen_list, mae_loss, "img_{}".format(cnt), processed_dir, best_model_output_dir, data_name)
            cnt += 1

            test_data.append(np.array(in_list))
            pre_data.append(np.array(gen_list))

            '''for i in range(len(test_batch)):
                test_data.append(test_batch[i].flatten().cpu().detach().numpy().T)
                pre_data.append(pre[i].flatten().cpu().detach().numpy().T)'''

        while numb < D:
            in_data, out_data = [], []
            for input_ba in tqdm(dataloaders['train']):
                input_ba = input_ba.float().to(DEVICE)
                optimizer.zero_grad()
                out, _ = loaded_model(input_ba)
                mae_loss = loss_fn(input_ba, out)
                in_list = [e[0] for e in input_ba.tolist()[0]]
                gen_list = [e[0] for e in out.tolist()[0]]
                in_list, gen_list = evaluate_loss_picture(in_list, gen_list, mae_loss, "img_{}".format(cnt), processed_dir, best_model_output_dir, data_name)
                cnt += 1

                for j in range(len(input_ba)):
                    in_data.append(np.array(input_ba[j].flatten().cpu().detach().tolist()).T)
                    out_data.append(np.array(out[j].flatten().cpu().detach().tolist()).T)
                    '''if data_name != 'Fatigue':
                        in_data.append(np.array([0] + input_ba[j].flatten().cpu().detach().tolist()).T)
                        out_data.append(np.array([0] + out[j].flatten().cpu().detach().tolist()).T)
                    else:
                        in_data.append(np.array(input_ba[j].flatten().cpu().detach().tolist()).T)
                        out_data.append(np.array(out[j].flatten().cpu().detach().tolist()).T)'''

            numb += 1
            constant = ori_data[:len(in_data), 0]
            in_data = sample_extend0(np.array(in_data), constant)
            out_data = sample_extend0(np.array(out_data), constant)

            data_list.append(np.array(in_data))
            out_list.append(np.array(out_data))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x_ticks = np.arange(0, input_ba.shape[1] + 1, 1)
    x = np.linspace(0, input_ba.shape[1], input_ba.shape[1] + 1)
    '''if data_name != 'Fatigue':
        x_ticks = np.arange(0, input_ba.shape[1] + 1, 1)
        x = np.linspace(0, input_ba.shape[1], input_ba.shape[1] + 1)
    else:
        x_ticks = np.arange(0, input_ba.shape[1], 1)
        x = np.linspace(0, input_ba.shape[1] - 1, input_ba.shape[1])'''

    #constant = np.zeros((len(test_data), 1))[:, 0]
    constant = ori_data[-len(test_data):, 0]
    test_data = sample_extend0(np.array(test_data), constant)
    pre_data = sample_extend0(np.array(pre_data), constant)

    od = ax.plot(x, np.array(test_data).T, 'bo--', label='Original')
    plt.setp(od[1:], label="_")

    sd = ax.plot(x, np.array(pre_data).T, 'm^--', label='SVAE-GRU')
    plt.setp(sd[1:], label="_")

    plt.xlabel("Time")
    plt.xticks(x_ticks)
    ax.legend()
    #plt.show()
    plt.savefig(best_model_output_dir + 'prediction.png')
    plt.cla()
    plt.close('all')

    return ori, np.array(test_data), np.array(pre_data), np.array(data_list), np.array(out_list)


def loss_picture(in_list, gen_list, mse_loss, name, data_name):
    # loss
    in_list = in_list.squeeze(-1).tolist()
    gen_list = gen_list.squeeze(-1).tolist()
    '''if data_name != 'Fatigue':
        in_list = [0] + in_list.squeeze(-1).tolist()
        gen_list = [0] + gen_list.squeeze(-1).tolist()
    else:
        in_list = in_list.squeeze(-1).tolist()
        gen_list = gen_list.squeeze(-1).tolist()'''
    plt.plot(in_list, label='input', color='b')
    plt.plot(gen_list, label='generator', color='r')
    plt.title('{} with loss {}'.format(name, mse_loss), fontdict=font)
    plt.xlabel('Time step', fontdict=font)
    plt.ylabel('Value', fontdict=font)

    plt.subplots_adjust(left=0.15)
    # plt.show()
    # plt.savefig('{}.png'.format(name))


def evaluate_loss_picture(in_list, gen_list, mse_loss, name, processed_dir, best_model_output_dir, data_name):
    min = torch.load(os.path.join(processed_dir, 'min.pt')).tolist()
    max = torch.load(os.path.join(processed_dir, 'max.pt')).tolist()

    for i, e in enumerate(in_list):
        in_list[i] = e * (max[i] - min[i] + 1e-5) + min[i]
        gen_list[i] = gen_list[i] * (max[i] - min[i] + 1e-5) + min[i]

    # loss
    '''if data_name != 'Fatigue':
        in_list = [0] + in_list
        gen_list = [0] + gen_list'''
    plt.plot(in_list, label='input', color='b')
    plt.plot(gen_list, label='generator', color='r')
    plt.title('{} with loss {}'.format(name, mse_loss), fontdict=font)
    plt.xlabel('Time step', fontdict=font)
    plt.ylabel('Value', fontdict=font)

    plt.subplots_adjust(left=0.15)
    # plt.show()
    plt.savefig(best_model_output_dir + '{}.png'.format(name))
    return in_list, gen_list


if __name__ == '__main__':
    import numpy as np
    import torch
    from numpy import random
    from scipy.stats import geninvgauss

    def setup_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    setup_seed(2023)

    def inverse_gaussian_process(Time, N, alpha, beta):

        dt = Time / N

        inverse_gaussian_values = geninvgauss.rvs(alpha * dt, beta * dt ** 2, size=N) / 50
        inverse_gaussian_process = np.cumsum(inverse_gaussian_values)
        inverse_gaussian_process = np.insert(inverse_gaussian_process, 0, 0.0)
        return inverse_gaussian_process

    Time = 20
    N = 20
    sample_num = 20  # 00
    alpha0 = 2.0
    beta0 = 1.0
    ori_data = []
    D = 10 **3

    sample_cnt = 0
    for i in range(sample_num):
        ori_data.append(inverse_gaussian_process(Time, N, alpha0, beta0))
        sample_cnt += 1
        if sample_cnt % 10 == 0:
            print("generate {} total {}".format(sample_cnt, sample_num))
    ori_data = np.array(ori_data)
    n_epoch = 200
    ori, svae_gru_data, svae_gru_pre, svae_gru_in, svae_gru_out = svae_gru(ori_data, D, n_epoch)
