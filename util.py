from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def showConfusionMatrix(model, x, y labels = ['healthy', 'multiple_diseases', 'rust', 'scab'])
    
    y_pred = np.argmax(model.predict(x), axis=1)
    mat = confusion_matrix(y, y_pred)

    df_cm = pd.DataFrame(mat, index = [i for i in labels],  columns = [i for i in labels])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)