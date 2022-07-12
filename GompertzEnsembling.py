snap0 = tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/models_cycle_0.h5')
snap0.evaluate(X_test, y_test)
snap1 = tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/models_cycle_1.h5')
snap1.evaluate(X_test, y_test)
snap2 = tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/models_cycle_2.h5')
snap2.evaluate(X_test, y_test)
snap3 = tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/models_cycle_3.h5')
snap3.evaluate(X_test, y_test)
snap4 = tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/models_cycle_4.h5')
snap4.evaluate(X_test, y_test)
models = [snap0, snap1, snap2, snap3, snap4]
predsf5 = [model.predict(X_test) for model in models]
predsf5 = np.array(predsf5)
s0p=snap0.predict(X_test)
s1p=snap1.predict(X_test)
s2p=snap2.predict(X_test)
s3p=snap3.predict(X_test)
s4p=snap4.predict(X_test)
snap5=tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/continued_SS/cSS_models_cycle_0.h5')
print(snap5.evaluate(X_test, y_test))
snap6=tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/continued_SS/cSS_models_cycle_1.h5')
print(snap6.evaluate(X_test, y_test))
snap7=tf.keras.models.load_model('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3/continued_SS/cSS_models_cycle_2.h5')
print(snap7.evaluate(X_test, y_test))
s5p = snap5.predict(X_test)
s6p = snap6.predict(X_test)
s7p = snap7.predict(X_test)
y_test_1d = encode_y(y)
def Gompertz(top = 2, *argv):
    L = 0 #Number of classifiers
    for arg in argv:
        L += 1

    num_classes = arg.shape[1]
    CF = np.zeros(shape = (L,arg.shape[0], arg.shape[1]))

    for i, arg in enumerate(argv):
        CF[:][:][i] = arg

    R_L = fuzzy_rank(CF, top) #R_L is with penalties
    
    RS = np.sum(R_L, axis=0)
    CFS = CFS_func(CF, R_L)
    FS = RS*CFS

    predictions = np.argmin(FS,axis=1)
    return predictions
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test_1d, ensemble_prediction)
print(cm2)
ensembled_accuracy_l3_avg = accuracy_score(ensemble_prediction_l3_avg, y_test_1d)
ensembled_accuracy_l3_avg
