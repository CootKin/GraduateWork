import model

network = model.TestModel()
network.compile_model('binary_crossentropy', 'adam', ['accuracy'])
network.model.summary()
network.save_model("test.h5")

network2 = model.TestModel()
network2.load_model("test.h5")
network2.model.summary()