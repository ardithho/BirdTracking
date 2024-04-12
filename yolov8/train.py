import yaml
from ultralytics import YOLO
from roboflow import Roboflow


def load_dataset_head(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-head', version=3):
    """
    Load the dataset for feat recognition.

    Parameters:
    - api_key (str): The API key for accessing the Roboflow API.
    - workspace (str): The name of the workspace.
    - project (str): The name of the project.
    - version (int): The version of the project.

    Returns:
    - dataset (str): The downloaded dataset.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    return project.version(version).download('yolov8', f'datasets/{project_name}-{version}')


def load_dataset_feat(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-feature-detailed', version=9):
    """
    Load the dataset for feat recognition.

    Parameters:
    - api_key (str): The API key for accessing the Roboflow API.
    - workspace (str): The name of the workspace.
    - project (str): The name of the project.
    - version (int): The version of the project.

    Returns:
    - dataset (str): The downloaded dataset.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    return project.version(version).download('yolov8', f'datasets/{project_name}-{version}')


def train_model(data, model='yolov8n.yaml', epochs=200, batch=16, imgsz=416, name='train'):
    """
    Train the model for toy recognition.

    Parameters:
    - data (str): The dataset to train the model on.
    - model (str): The model to use. Default is "yolov8".
    - epochs (int): The number of epochs to train for. Default is 100.
    - batch (int): The batch size to use. Default is 16.
    - imgsz (int): The size of the images. Default is 416.
    - weights (str): The weights to use. Default is "yolov8.pt".

    Returns:
    - model (YOLO): The trained model.
    """
    model = YOLO(model)
    model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz, name=name)
    return model


if __name__ == '__main__':
    name = 'feat'
    dataset = None
    exec(f'dataset = load_dataset_{name}()')
    data_file = f'{dataset.location}/data.yaml'
    with open(data_file, 'r') as f:
        data = yaml.safe_load(f)
    data['train'] = '../train/images'
    data['val'] = '../valid/images'
    with open(data_file, 'w') as f:
        yaml.dump(data, f)
    model = train_model(data_file, batch=32, name=name)
