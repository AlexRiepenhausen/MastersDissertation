import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from word2vec.dataReaderDoc import DataReader, Word2vecDataset
from word2vec.word2vec import SkipGramModel

from utilities.utilities import weightInit

class Word2VecTrainer:
    def __init__(self, primary_files, supporting_files=None,
                 emb_dimension=10, batch_size=32, window_size=5, initial_lr=0.1, min_count=1):

        # the actual data
        self.data = DataReader(primary_files, min_count, supporting_files)

        # training hyperparameters
        self.emb_size       = len(self.data.word2id)
        self.emb_dimension  = emb_dimension
        self.window_size    = window_size
        self.batch_size     = batch_size
        self.initial_lr     = initial_lr
        self.iter_per_epoch = 0

        # init model
        self.skip_gram_model = SkipGramModel(self.emb_size, self.emb_dimension)
        self.device          = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # return string containing information about this training session
    def toString(self):
        return "lr_{}_bs_{}_ipe_{}_embs_{}_embd_{}_win_{}".format(self.initial_lr,
                                                                  self.batch_size,
                                                                  self.iter_per_epoch,
                                                                  self.emb_size,
                                                                  self.emb_dimension,
                                                                  self.window_size)


    # tell the data loader to iterate over a specific set of files
    def initDataLoader(self, training_files):
        dataset = Word2vecDataset(self.data, self.window_size, custom_files=training_files)
        return DataLoader(dataset, self.batch_size, shuffle=False, num_workers=0, collate_fn=dataset.collate)


    def initDevice(self):

        if torch.cuda.device_count() > 1:
            print("Available GPUs: ", torch.cuda.device_count())
            self.skip_gram_model = torch.nn.DataParallel(self.skip_gram_model)

        if torch.cuda.is_available():
            self.skip_gram_model.cuda()


    # initialise/refresh weights of model
    def weightInitialisation(self, init, saved_model_path=None):
        if init == weightInit.fromScratch:
            self.skip_gram_model.weight_init()
        if init == weightInit.load:
            self.skip_gram_model.load_state_dict(torch.load(saved_model_path))
            self.skip_gram_model.eval()
        if init == weightInit.inherit:
            pass  # inherit from previous training session of the same class -> do nothing


    # train word2vec model
    def train(self, training_files, output_file, num_epochs=100, init=weightInit.fromScratch, model_path=None):

        losses     = list()
        dataloader = self.initDataLoader(training_files)

        self.weightInitialisation(init, saved_model_path=model_path)
        self.initDevice()

        for iteration in tqdm(range(num_epochs)):

            optimizer = optim.SparseAdam(self.skip_gram_model.parameters(), lr=self.initial_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))

            count           = 0.0
            running_loss    = 0.0
            cumulative_loss = 0.0

            for i, sample_batched in enumerate(dataloader):

                if len(sample_batched[0]) > 1:
                    pos_u = sample_batched[0].to(self.device)
                    pos_v = sample_batched[1].to(self.device)
                    neg_v = sample_batched[2].to(self.device)

                    scheduler.step()
                    optimizer.zero_grad()
                    loss = self.skip_gram_model.forward(pos_u, pos_v, neg_v)
                    loss.backward()
                    optimizer.step()

                    running_loss = running_loss * 0.9 + loss.item() * 0.1
                    cumulative_loss += loss.item()

                    count += 1.0

            losses.append(cumulative_loss / count)
            self.iter_per_epoch = int(count * self.batch_size)

        self.skip_gram_model.module.save_embedding(self.data.id2word, output_file, self.data.max_num_words_file)

        return losses
