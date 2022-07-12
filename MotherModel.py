IMG_SHAPE = (256,256,3)
InceptionV3_model = tf.keras.applications.inception_v3.InceptionV3(input_shape= IMG_SHAPE, 
                                                                   include_top = False, 
                                                                   weights='imagenet')
def create_model(model_name):
  models = { "InceptionV3" : InceptionV3_model }
  model = models[model_name]
  for layer in model.layers:
    layer.trainable = True
  x = tf.keras.layers.Flatten()(model.output)
  x = tf.keras.layers.Dense(512, activation='relu')(x)
  x = tf.keras.layers.Dense(256,activation='relu')(x)
  x = tf.keras.layers.Dense(128,activation='relu')(x)
  x = tf.keras.layers.Dense(5,activation='softmax')(x)

  model = Model(inputs= model.input, outputs=x)

  
  my_model = tf.keras.models.clone_model(model)
  return my_model 


mothermodel1 = create_model("InceptionV3")

# Compile the model
mothermodel1.compile(loss='categorical_crossentropy',
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])
# Fit data to model
from datetime import datetime
from keras.callbacks import ModelCheckpoint

cbs = [Snapshot('/content/gdrive/MyDrive/Saved_Models/CervicalC/ADAM_SE_inception_V3', nb_epochs=125, verbose=2, nb_cycles=5)]


start = datetime.now()
epochs=125
history = mothermodel1.fit(x = train,
                          steps_per_epoch=203,
                          epochs=epochs,
                          validation_data=validation,
                          validation_steps=51,
                          verbose=2,
                          callbacks=[cbs]
                      )
duration = datetime.now() - start
print("Training completed in time: ", duration)
