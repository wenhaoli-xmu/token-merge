def get_modifier(method: str, model_type):

    if method == 'origin':
        from .origin import Origin
        return Origin

    elif method == 'greedy':
        from .greedy import Greedy
        return Greedy
    
    elif method == 'train':
        from .train import ModelForTraining
        return ModelForTraining

    elif method =='eval':
        from .eval import ModelForEvaluation
        return ModelForEvaluation
