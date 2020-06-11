import os
import re
val_rate = 0.2

# PREPARES DATA WITH CONTEXT (3 sentences per context for 1 response)

def main():
    data_dir = os.getcwd()
    text_dir = os.path.join(data_dir, 'en_super_short.txt')
    text = open(text_dir).read()
    lines = text.split('\n')
    contexts = []
    responses = []
    print('num lines: ' + str(len(lines)))
    for i, line in enumerate(lines):
        if (i < 10 or len(line) == 0):
            continue
        if (len(line.split(' ')) < 24):
            response = re.sub(r'[-|*"#^¦\\/~_¬]', '', line)
            context = ''
            for j in range(3):
                tmpLine = re.sub(r'[-|*"#^¦\\/~_¬]', '', lines[i - 3 + j])
                context += ' ' + tmpLine
            if (re.search(r'[:♪]', response) or re.search(r'[:♪]', context)):
                continue
            contexts.append(context)
            responses.append(response)

    if (len(contexts) != len(responses)):
        print('lengths of contexts and responses do not match!')
        exit(1)

    train_x_file = open(os.path.join(data_dir, 'train_x.txt'), mode='w+')
    val_x_file = open(os.path.join(data_dir, 'val_x.txt'), mode='w+')
    train_y_file = open(os.path.join(data_dir, 'train_y.txt'), mode='w+')
    val_y_file = open(os.path.join(data_dir, 'val_y.txt'), mode='w+')
    for i in range(int(len(contexts) * (1 - val_rate))):
        train_x_file.write(contexts[i] + '\n')
        train_y_file.write(responses[i] + '\n')
    for i in range(int(len(contexts) * (1 - val_rate)), len(contexts)):
        val_x_file.write(contexts[i] + '\n')
        val_y_file.write(responses[i] + '\n')


if __name__ == '__main__':
    main()



