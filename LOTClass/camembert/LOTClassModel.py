import os
import shutil
from sklearn.metrics import classification_report
from pathlib import Path
from LOTClass.camembert.trainer import LOTClassTrainer
from LOTClass.camembert.config import LOTClassConfig


class LOTClassifier():
    def __init__(self, path, labels, preatreatement_fn=lambda x: x, args=LOTClassConfig()):
        # Check format of labels
        try:
            _ = labels[0][0][0]
            dim = 1
        except TypeError:
            dim = 0

        if os.path.exists(path) and not args.overwrite_dataset:
            raise Exception(f"dataset path '{path}' already exists.")
        
        if os.path.exists(path) and args.overwrite_dataset:
            shutil.rmtree(path)

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
        self.labels = labels
        self.preatreatement_fn = preatreatement_fn
        self.args = args # Should be a LOTClassConfig instance

    def build_test_dataset(self, X_test, y_test):
        if X_test is None or y_test is None:
            raise Exception("Either X_test or y_test (or both) are undefined.") 

        x_test_path = os.path.join(self.path, self.args.test_file)
        y_test_path = os.path.join(self.path, self.args.test_label_file)

        if self.args.overwrite_dataset:
            if os.path.exists(x_test_path): os.remove(x_test_path)
            if os.path.exists(y_test_path): os.remove(y_test_path)

        # Write the test dataset to args.test_file:
        with open(x_test_path, 'a') as f:
            for x in X_test:
                f.write(self.preatreatement_fn(x) + '\n')

        # Write the labels (ground truth) to args.test_label_file
        with open(y_test_path, 'a') as f:
            for y in y_test:
                idx = 0
                for label_voc in self.labels:
                    if y in label_voc: break 
                    idx += 1
                f.write(str(idx) + '\n')

        self.trainer = None

    def fit(self, X, y=None):
        for text in X:
            text = text.replace(r'\n',  ' ')

            with open(os.path.join(self.path, "train.txt"), 'a') as f:
                f.write(self.preatreatement_fn(text) + '\n')
        
        self.trainer = LOTClassTrainer(self.args)
        # Construct category vocabulary
        self.trainer.category_vocabulary(top_pred_num=self.args.top_pred_num, 
                                    category_vocab_size=self.args.category_vocab_size)
        # Training with masked category prediction
        self.trainer.mcp(top_pred_num=self.args.top_pred_num, 
                    match_threshold=self.args.match_threshold, 
                    epochs=self.args.mcp_epochs)
        # Self-training 
        self.trainer.self_train(epochs=self.args.self_train_epochs, 
                           loader_name=self.args.final_model)
        # Write test set results
        if self.args.test_file is not None:
            self.trainer.write_results(loader_name=self.args.final_model, out_file=self.args.out_file)

    def classification_report(self, y_test):
        if self.trainer is None:
            raise Exception("Cannot output a classification report prior to training.\
                             Please run model.fit() first.")

        if not isinstance(y_test, list):
            raise Exception(f"Argument 'y_test' should be a list (got type '{type(y_test)}'")

        with open(os.path.join(self.path, self.args.out_file)) as file:
            y_pred = file.readlines()
            y_pred = [int(line.rstrip()) for line in y_pred]

        target_names = list(" ".join(v) for v in self.trainer.label_name_dict.values())

        numeric_y_test = []
        for label_words in y_test:
            for key, value in self.trainer.label_name_dict.items():
                if any([label.upper() in map(str.upper, value) for label in label_words.split()]):
                    numeric_y_test.append(key)
                    break

        assert len(numeric_y_test) == len(y_pred), f"Make sure the names of the labels in the test file match the labels of the model."

        print(classification_report(numeric_y_test, y_pred, target_names=target_names))
        
        return numeric_y_test, y_pred
