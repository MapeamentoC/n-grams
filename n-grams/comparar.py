import glob


for file in glob.glob('./Utilizados\\*.txt'):
    second_file = file.replace('./Utilizados\\', '.\\utilitarios\\')
    out_path = file.replace('./Utilizados\\', '.\\Comparados\\')
    with open(file, 'r', encoding='utf8') as th_file:
        with open(second_file, 'r', encoding='utf8') as sd_file:
            base = sd_file.readlines()
            with open(out_path, 'w', encoding='utf8') as outfile:
                outfile.write('FICOU')
                outfile.write('\t')
                outfile.write('SAIU')
                outfile.write('\n')
                for line in th_file:
                    if line in base:
                        outfile.write(line.split('\t')[0])
                        outfile.write('\t')
                        outfile.write('#')
                        outfile.write('\n')
                    else:
                        outfile.write('#')
                        outfile.write('\t')
                        outfile.write(line.split('\t')[0])
                        outfile.write('\n')
