import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import f1_score, confusion_matrix, classification_report

#from train import Trainer


def evaluate_batch(dataloader, model_, device_, model_path):
    model_.load_state_dict(torch.load(model_path))
    model_ = model_.to(device_)
    model_.eval()
    labels_list, preds_list = [], []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            inputs = batch.inputs_word.to(device_)
            labels = batch.labels.to(device_)

            outputs = model_(inputs)
            labels_list.extend(labels.cpu().detach().numpy().tolist())
            preds_list.extend(outputs[1].cpu().detach().numpy().tolist())
    return preds_list, labels_list


def calculate_f1_score(preds_list, labels_list):
    y_true = [item for sublist in labels_list for item in sublist]
    y_pred = [item for sublist in preds_list for item in sublist]
    return f1_score(y_true, y_pred, average='weighted')


def create_classification_report(preds_list, labels_list):
    y_true = [item for sublist in labels_list for item in sublist]
    y_pred = [item for sublist in preds_list for item in sublist]
    return classification_report(y_pred, y_true, target_names=['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG',
                                                               'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC'])


def create_confusion_matrix(preds_list, labels_list):
    y_true = [item for sublist in labels_list for item in sublist]
    y_pred = [item for sublist in preds_list for item in sublist]
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='g')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    tick_list = ['<pad>', '<bos>', '<eos>', 'O', 'I-PER', 'I-ORG', 'I-LOC', 'I-MISC', 'B-MISC', 'B-ORG', 'B-LOC']
    ax.xaxis.set_ticklabels(tick_list)
    ax.yaxis.set_ticklabels(tick_list)
    plt.savefig('images/confusion_matrix.png')


if __name__ == '__main__':
    trainer = Trainer()
    preds_list, labels_list = evaluate_batch(trainer.data['iterators'][2], trainer.model,
                                       trainer.device, trainer.model_saving_path)
    print(calculate_f1_score(preds_list, labels_list))
    print(create_classification_report(preds_list, labels_list))
    create_confusion_matrix(preds_list, labels_list)


