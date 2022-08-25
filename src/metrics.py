from sklearn.metrics import f1_score

def get_f1_score(pred, true):
    pred = pred.argmax(1).cpu().detach().numpy().reshape(-1)
    true = true.cpu().detach().numpy().reshape(-1)
    return f1_score(true, pred, average="macro")


def get_filtred_accuracy_score(pred, true):
    pred = pred.argmax(1).cpu().detach().numpy().reshape(-1)
    true = true.cpu().detach().numpy().reshape(-1)
    mask = ((pred == 1) & (true == 1)) | ((pred == 0) & (true == 0))
    return accuracy_score(pred[~mask], true[~mask])