import matplotlib.pyplot as plt

def plotHistory(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label = 'train_accuracy')
    plt.plot(epochs, val_acc, 'b', label = 'val_accuracy')
    plt.title('Train acc and Val acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.figure()


    plt.plot(epochs, acc, 'r', label = 'train_loss')
    plt.plot(epochs, acc, 'b', label = 'train_loss')
    plt.title('Train loss and Val loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()