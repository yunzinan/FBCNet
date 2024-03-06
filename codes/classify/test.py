import fbcnet


if __name__ == "__main__":
    fbcnet = fbcnet.FBCNet()

    fbcnet.loadModel()

    fbcnet.finetune()

    fbcnet.inference()
    