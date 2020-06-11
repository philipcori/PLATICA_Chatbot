import os
import re
val_rate = 0.0001

def main():
    data_dir = os.getcwd()
    text_dir = os.path.join(data_dir, 'en_super_short.txt')
    text = open(text_dir).read()
    lines = text.split('\n')
    statements = []
    questions = []
    for i, line in enumerate(lines):
        if (i == 0 or len(line) == 0):
            continue
        if (len(line.split(' ')) < 24 and 
                not re.search(r'[:♪\]\[\(\)]', line) and 
                not re.search(r'[:♪\]\[\(\)]', lines[i - 1])):
            currLine = re.sub(r'[-|*"#^¦\\/~_¬]', '', line)
            lastLine = re.sub(r'[-|*"#^¦\\/~_¬]', '', lines[i - 1])
            statements.append(lastLine)
            questions.append(currLine)

    if (len(statements) != len(questions)):
        print('lengths of statements and questions do not match!')
        exit(1)

    print(str(len(statements)) + ' lines kept out of ' + str(len(lines)) + '. -- ' + str(len(statements) / len(lines)))
    train_x_file = open(os.path.join(data_dir, 'train_x.txt'), mode='w+')
    val_x_file = open(os.path.join(data_dir, 'val_x.txt'), mode='w+')
    train_y_file = open(os.path.join(data_dir, 'train_y.txt'), mode='w+')
    val_y_file = open(os.path.join(data_dir, 'val_y.txt'), mode='w+')
    for i in range(int(len(statements) * (1 - val_rate))):
        train_x_file.write(statements[i] + '\n')
        train_y_file.write(questions[i] + '\n')
    for i in range(int(len(statements) * (1 - val_rate)), len(statements)):
        val_x_file.write(statements[i] + '\n')
        val_y_file.write(questions[i] + '\n')


if __name__ == '__main__':
    main()



