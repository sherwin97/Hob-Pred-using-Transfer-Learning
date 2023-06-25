import torch
import torch.nn as nn
from sklearn.metrics import r2_score, accuracy_score, f1_score, roc_auc_score


class EngineSol:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.MSELoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1), outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()

        return final_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()
        return final_loss / len(data_loader)

    def test(self, data_loader):
        self.model.eval()
        final_loss = 0
        r2_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()

                r2 = r2_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    outputs.to("cpu").detach().numpy(),
                )
                r2_total += r2

        mse = final_loss / len(data_loader)

        return mse, r2_total / len(data_loader)


class EngineHOB:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_attr, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1), outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        acc_total = 0
        f1_total = 0
        roc_auc_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()
                acc = accuracy_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                acc_total += acc

                f1 = f1_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                f1_total += f1

                roc_auc = roc_auc_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.sigmoid(outputs).to("cpu").detach().numpy(),
                )
                roc_auc_total += roc_auc

        bce = final_loss / len(data_loader)
        acc_score = acc_total / len(data_loader)
        f1score = f1_total / len(data_loader)
        rocauc = roc_auc_total / len(data_loader)
         
        return bce, acc_score, f1score, rocauc

    def test(self, data_loader):
        self.model.eval()
        final_loss = 0
        acc_total = 0
        f1_total = 0
        roc_auc_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                )
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()

                acc = accuracy_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                acc_total += acc

                f1 = f1_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                f1_total += f1

                roc_auc = roc_auc_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.sigmoid(outputs).to("cpu").detach().numpy(),
                )
                roc_auc_total += roc_auc

        bce = final_loss / len(data_loader)
        acc_score = acc_total / len(data_loader)
        f1score = f1_total / len(data_loader)

        return bce, acc_score, f1score, roc_auc_total / len(data_loader)
    
    def get_embeds(self, data_loader):
        embeddings_list = []
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs, embeddings = self.model(
                    data.x, data.edge_attr, data.edge_index, data.batch
                        )
                embeddings_list.append(embeddings)
        return embeddings_list
                


class EngineHOB_no_edge:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.device = device
        self.optimizer = optimizer

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data.x, data.edge_index, data.batch)
            loss = self.loss_fn(data.y.unsqueeze(1), outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
        return final_loss / len(data_loader)

    def validate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()
        return final_loss / len(data_loader)

    def test(self, data_loader):
        self.model.eval()
        final_loss = 0
        acc_total = 0
        f1_total = 0
        roc_auc_total = 0
        with torch.no_grad():
            for data in data_loader:
                data = data.to(self.device)
                outputs = self.model(data.x, data.edge_index, data.batch)
                loss = self.loss_fn(data.y.unsqueeze(1), outputs)
                final_loss += loss.item()

                acc = accuracy_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                acc_total += acc

                f1 = f1_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.round(torch.sigmoid(outputs)).to("cpu").detach().numpy(),
                )
                f1_total += f1

                roc_auc = roc_auc_score(
                    data.y.unsqueeze(1).to("cpu").detach().numpy(),
                    torch.sigmoid(outputs).to("cpu").detach().numpy(),
                )
                roc_auc_total += roc_auc

        return (
            final_loss / len(data_loader),
            acc_total / len(data_loader),
            f1_total / len(data_loader),
            roc_auc_total / len(data_loader),
        )
