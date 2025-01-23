class Config:
    def __init__(self):
        '''
        model name mapping class:
            map from a short name to a longer and formal name.
        '''
        self.models = {
                'gpt3': 'gpt3',
                'gpt2-large': 'gpt2-large',
                
                'roberta-base': 'roberta-base',
                'roberta-large': 'roberta-large',
                
                't5-small': 't5-small', 
                't5-base': 't5-base', 
                't5-large': 't5-large', 
                't5-3b': 't5-3b', 
                't5-11b': 't5-11b', 
        }
