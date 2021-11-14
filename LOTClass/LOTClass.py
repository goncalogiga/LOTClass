import os
from pathlib import Path
from LOTClass.trainer import LOTClassTrainer
from LOTClass.config import LOTClassConfig


class LOTClass():
    def __init__(self, path, labels, preatreatement_fn=lambda x: x, args=LOTClassConfig()):
        # Check format of labels
        try:
            _ = labels[0][0]
            dim = 1
        except TypeError:
            dim = 0

        if os.path.exists(path):
            raise Exception(f"dataset path '{path}' already exists.")

        os.mkdir(path)

        args.dataset_dir = path

        with open(os.path.join(path, "label_names.txt"), 'a') as f:
            for label in labels:
                if dim == 1:
                    label = [preatreatement_fn(l) for l in label]
                    f.write(" ".join(label) + '\n')
                else:
                    f.write(preatreatement_fn(label) + '\n')

        self.path = path
        self.preatreatement_fn = preatreatement_fn
        self.args = args # Should be a LOTClassConfig instance

    def fit(self, X, y=None):
        for text in X:
            text = text.replace(r'\n',  ' ')

            with open(os.path.join(self.path, "train.txt"), 'a') as f:
                f.write(self.preatreatement_fn(text) + '\n')
        
        trainer = LOTClassTrainer(self.args)
        # Construct category vocabulary
        trainer.category_vocabulary(top_pred_num=self.args.top_pred_num, 
                                    category_vocab_size=self.args.category_vocab_size)
        # Training with masked category prediction
        trainer.mcp(top_pred_num=self.args.top_pred_num, 
                    match_threshold=self.args.match_threshold, 
                    epochs=self.args.mcp_epochs)
        # Self-training 
        trainer.self_train(epochs=self.args.self_train_epochs, 
                           loader_name=self.args.final_model)
