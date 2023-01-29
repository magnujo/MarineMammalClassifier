from MLModels import architectures as arc

m = arc.convnext((224, 224, 3), 31)
print(m.summary())
print("DONE")







