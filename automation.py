import sys
import fileinput

source = 'ui/qsom.py'
import_start_line_no = 1
replaces_list = [('QtGui.QLineEdit', 'ParamLineEdit',)]
import_line = 'from ui.uiutils import ParamLineEdit'


def insert_line(act_pos, isrt_pos, linep):
    ''' inserts lines (strings) indicating imports into file '''
    if act_pos == isrt_pos:
        sys.stdout.write(linep)
        sys.stdout.write('\n')


def replace_text(replacep, line):
    ''' replaces first element of (touple) pain by the second one '''
    for oldp, newp in replacep:
        sys.stdout.write(line.replace(oldp, newp))


def automate_correction(ipos, iline, replacep):
    ''' automates file correction

    :param ipos: position (line number) where we want to place imports that should be added
    :param iline: string containing definition of imports separated by '\n' as it breaks line
    :param replacep: list of touples containing two parameters (more -> replace_text doc)
    :return:
    '''
    for i, line in enumerate(fileinput.input(source, inplace=True)):
        insert_line(i, ipos, iline)
        replace_text(replacep, line)

if __name__ == '__main__':
    print('running automation...')
    automate_correction(9, import_line, replaces_list)
    print('automation done')