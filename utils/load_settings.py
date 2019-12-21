import yaml 


def load_settings():
    with open("utils/settings.yml") as f:
        params = yaml.safe_load(f)
    return params
