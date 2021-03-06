import tensorflow as tf
import tensorflow.keras.backend as K

class DiceCoefficient(tf.keras.metrics.Metric):    
    def __init__(self, name='dice_coefficient', **kwargs):
        super(DiceCoefficient, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='dice_coeff', initializer='zeros')
        

    def update_state(self, y_true, y_pred,  smooth=1.,eps: float = 1e-9):
        y_true=zer_one(y_true)
        y_pred=zer_one(y_pred)
        batch_num=y_true.shape[0]
        #print("batch_num",batch_num)

        #print("y_true",y_true.shape,"y_pred",y_pred.shape)
        intersec = K.sum(y_true * y_pred,axis=[1,2,3,4])
        union = K.sum(y_pred ,axis=[1,2,3,4]) + K.sum(y_true,axis=[1,2,3,4]) 
        #print("intersec",intersec,"union",union)
        dice_coefficient=(2*intersec + eps) / (union+eps)
        dice_coefficient_arr=dice_coefficient.numpy()
        #print("dice_coefficient_arr shape",dice_coefficient_arr.shape)
        for i in range(len(dice_coefficient_arr)):
            if K.sum(y_true)==0 and K.sum(y_pred)==0:
                dice_coefficient_arr[i]=1
        dice_coefficient=tf.convert_to_tensor(dice_coefficient_arr)
        
        self.true_positives.assign_add(K.mean(dice_coefficient ))

    def result(self):
        return self.true_positives
    
def zer_one(img,threshold=0.5):
    ones = K.ones_like(img)    
    zeros = K.zeros_like(img)
    output = K.switch(K.greater(img,threshold), ones, img)
    output = K.switch(K.less_equal(output,threshold), zeros, output)
    return output

def confusion_matrix_calc(targets,inputs):
    inputs= tf.math.argmax(inputs, axis=-1)
    inputs=tf.expand_dims(inputs, -1, name=None)
    inputs=tf.cast(inputs, dtype=tf.float32, name=None)
    tn = K.sum(inputs * targets) #K.sum(inputs+targets-(inputs * targets))
    tp = K.sum((inputs * targets))
    fp = K.sum(((1-targets) * inputs))
    fn = K.sum((targets * (1-inputs)))
    return tn,fp,fn,tp

BETA=0.5
ALPHA=1.5
threshold=0.5
 
#@tf.function
def TPR2(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    tn, fp, fn, tp=confusion_matrix_calc(targets,inputs)
    TPR = (tp +smooth)/(tp+fn+smooth)
    return TPR
#@tf.function

def TPR_FPR(targets, inputs, smooth=1e-6):
    inputs = zer_one(inputs)
    tp = tf.keras.metrics.TruePositives()
    tp.update_state(targets, inputs)
    tp=tp.result().numpy()
    
    fp = tf.keras.metrics.FalsePositives()
    fp.update_state(targets, inputs)
    fp=fp.result().numpy()
    
    fn = tf.keras.metrics.FalseNegatives()
    fn.update_state(targets, inputs)
    fn=fn.result().numpy()
    
    tn = tf.keras.metrics.TrueNegatives()
    tn.update_state(targets, inputs)
    tn=tn.result().numpy()
    
    #fnr FN/FN+TP
    #tpr TP/TP+FN
    #tnr TN/TN+FP
    #ppv TP/TP+FP.
    #fpr FP/FP+TN
    # FPR = FP/FP+TP valverde
    # VD = |TPs ???TPgt|/TPgt  valverde

    return tp/(tp+fn), fp/(fp+tn)


def FPR(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    tn, fp, fn, tp=confusion_matrix_calc(targets,inputs)
    FPR = (fp +smooth)/(fp+tp+ smooth) 
    return FPR
#@tf.function 
def FNR(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    tn, fp, fn, tp=confusion_matrix_calc(targets,inputs)
    FNR = (fn+smooth)/(tp+fn+smooth)
    return FNR
      
def Tversky(targets, inputs, alpha=ALPHA, beta=BETA, smooth=1e-6):
    tn, fp, fn, tp=confusion_matrix_calc(targets,inputs)
        
    #tn, fp, fn, tp=confusion_matrix_calc(targets,inputs).ravel()
    Tversky = (tp + smooth) / (tp + alpha*fp + beta*fn + smooth) 
    return Tversky

def dice_coefficient(y_true, y_pred, smooth=1.,eps: float = 1e-9):
    y_true=zer_one(y_true)
    y_pred=zer_one(y_pred)
    batch_num=y_true.shape[0]
    #print("batch_num",batch_num)

    #print("y_true",y_true.shape,"y_pred",y_pred.shape)
    intersec = K.sum(y_true * y_pred,axis=[1,2,3,4])
    union = K.sum(y_pred ,axis=[1,2,3,4]) + K.sum(y_true,axis=[1,2,3,4]) 
    #print("intersec",intersec,"union",union)
    dice_coefficient=(2*intersec + eps) / (union+eps)
    dice_coefficient_arr=dice_coefficient.numpy()
    #print("dice_coefficient_arr shape",dice_coefficient_arr.shape)
    for i in range(len(dice_coefficient_arr)):
        if K.sum(y_true)==0 and K.sum(y_pred)==0:
            dice_coefficient_arr[i]=1
    dice_coefficient=tf.convert_to_tensor(dice_coefficient_arr)
    return K.mean(dice_coefficient )

def MeanIoU(y_true, y_predict):
    y_predict= zer_one(y_predict)
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_true, y_predict)
    return m.result().numpy()