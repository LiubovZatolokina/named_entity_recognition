import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_preparation import conll2003_dataset
from evaluate import evaluate_batch, calculate_f1_score
from ner_model import BiLSTM_CRF


class Trainer(nn.Module):
    def __init__(self):
        super(Trainer, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.epoch_number = 300
        self.model_saving_path = './ner_model.pt'
        self.tag_to_ix = {'<unk>': 0, '<pad>': 1, '<bos>': 2, '<eos>': 3, 'O': 4, 'I-PER': 5, 'I-ORG': 6, 'I-LOC': 7,
                          'I-MISC': 8,
                          'B-MISC': 9, 'B-ORG': 10, 'B-LOC': 11}
        self.batch_size = 64
        self.embedding_size = 128
        self.hidden_size = 256
        self.data = conll2003_dataset('ner', self.batch_size)
        self.model = BiLSTM_CRF(len(self.data['vocabs'][0]), self.tag_to_ix, self.embedding_size, self.hidden_size)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)


trainer = Trainer()


def train():
    tb = SummaryWriter()
    for epoch in tqdm(range(trainer.epoch_number)):
        loss_dict = {}
        train_loss = 0.0
        valid_loss = 0.0
        for idx, batch in enumerate(trainer.data['iterators'][0]):
            trainer.model.train()
            trainer.model.zero_grad()
            with torch.set_grad_enabled(True):
                loss = trainer.model.neg_log_likelihood(batch.inputs_word.to(trainer.device),
                                                        batch.labels.to(trainer.device))
                loss.backward()
                trainer.optimizer.step()
                train_loss += loss.item() * batch.inputs_word.size(0)
        for idx, batch in enumerate(trainer.data['iterators'][1]):
            trainer.model.eval()
            trainer.model.zero_grad()
            with torch.set_grad_enabled(True):
                loss = trainer.model.neg_log_likelihood(batch.inputs_word, batch.labels)
                valid_loss += loss.item() * batch.inputs_word.size(0)

        torch.save(trainer.model.state_dict(), trainer.model_saving_path)
        preds_list, labels_list = evaluate_batch(trainer.data['iterators'][2], trainer.model,
                                                 trainer.device, trainer.model_saving_path)
        loss_dict['train'] = train_loss / len(trainer.data['iterators'][0].dataset)
        loss_dict['valid'] = valid_loss / len(trainer.data['iterators'][0].dataset)
        f1_score = calculate_f1_score(preds_list, labels_list)
        print('Loss', loss_dict)
        print('F1 score', f1_score)
        tb.add_scalars('Loss: epoch', {'Train': loss_dict['train'], 'Valid': loss_dict['valid']}, epoch)
        tb.add_scalars('F1 score: epoch', {'Test': f1_score}, epoch)


if __name__ == '__main__':
    train()
