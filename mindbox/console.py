def log(msg='', debug=True, flush=False, symbol=True):
    if (debug==True):
        if (flush==True):
            if (symbol == True):
                print('>> '+msg, end='\r', flush=flush)
            else:
                print(msg, end='\r', flush=flush)
        else:
            if (symbol == True):
                print('>> '+msg)
            else:
                print(msg)