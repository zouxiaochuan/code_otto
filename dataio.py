import constants

def write_result(result, filename):
    with open(filename, 'w') as f:
        f.write('session_type,labels\n')

        for r in result:
            for i, n in constants.etype_map_reverse.items():
                aids = ' '.join([str(aid) for aid in r[i + 1]])
                f.write(f'{r[0]}_{n},{aids}\n')
                pass
            pass
        pass
    pass