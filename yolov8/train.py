import yaml
from ultralytics import YOLO
from roboflow import Roboflow


def load_dataset(api_key='jyKUZIKA3yySfSqRdXqI',
                 workspace='bird-tracking-yvxlp',
                 project_name='', version=1):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    return project.version(version).download('yolov8', f'datasets/{project_name}-{version}')



def load_dataset_full(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-full', version=1):
    """
    Load the dataset for full detection.
    """
    return load_dataset(api_key=api_key, workspace=workspace,
                        project_name=project_name, version=version)


def load_dataset_head(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-head', version=3):
    """
    Load the dataset for head detection.
    """
    return load_dataset(api_key=api_key, workspace=workspace,
                        project_name=project_name, version=version)


def load_dataset_feat(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-feature-detailed', version=10):
    """
    Load the dataset for feat detection.
    """
    return load_dataset(api_key=api_key, workspace=workspace,
                        project_name=project_name, version=version)


def load_dataset_liner(api_key='jyKUZIKA3yySfSqRdXqI',
                       workspace='bird-tracking-yvxlp',
                       project_name='bird-feature-liner', version=2):
    """
    Load the dataset for liner detection.
    """
    return load_dataset(api_key=api_key, workspace=workspace,
                        project_name=project_name, version=version)


def load_dataset_pose(api_key='jyKUZIKA3yySfSqRdXqI',
                      workspace='bird-tracking-yvxlp',
                      project_name='bird-keypoints', version=1):
    """
    Load the dataset for pose estimation.
    """
    return load_dataset(api_key=api_key, workspace=workspace,
                        project_name=project_name, version=version)


def train_model(data, model='yolov8s.yaml', epochs=100, batch=16, imgsz=640, name='train'):
    """
    Train the model for object detection.

    Parameters:
    - data (str): The dataset to train the model on.
    - model (str): The model to use. Default is "yolov8s".
    - epochs (int): The number of epochs to train for. Default is 100.
    - batch (int): The batch size to use. Default is 16.
    - imgsz (int): The size of the images. Default is 640.
    - weights (str): The weights to use. Default is "yolov8s.pt".

    Returns:
    - model (YOLO): The trained model.
    """
    model = YOLO(model)
    model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz, name=name)
    return model


if __name__ == '__main__':
    name = 'liner'
    model = 'yolov8s-pose.yaml' if name == 'pose' else 'yolov8s.yaml'
    dataset = None
    exec(f'dataset = load_dataset_{name}()')
    data_file = f'{dataset.location}/data.yaml'
    with open(data_file, 'r') as f:
        data = yaml.safe_load(f)
    data['train'] = '../train/images'
    data['val'] = '../valid/images'
    with open(data_file, 'w') as f:
        yaml.dump(data, f)
    model = train_model(data_file, model, batch=32, imgsz=256, name=name, epochs=400)
