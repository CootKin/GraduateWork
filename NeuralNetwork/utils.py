import os
import shutil
import copy

from music21 import converter, note, chord, percussion, stream, bar, metadata
import music21
import numpy as np
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self):
        self.__path = os.path.join("./RefactorData/data")

    #def load_music(data_name, filename, n_bars, n_steps_per_bar):


class DataConverter:
    def __init__(self, directory):
        self.__data_path = os.path.join(f"./datasets/{directory}")
        self.__temp_path = os.path.join("./datasets/RefactorData/temp")
        self.__preprocessed_data_path = os.path.join("./datasets/RefactorData/data")

    # Извлекает все midi файлы во временную директорию
    def extract_files(self):
        directories = os.listdir(self.__data_path)
        dir_name = os.path.join(self.__data_path, directories[0])
        count_dirs = 10 # !Удалить
        for i, _dir in enumerate(directories): # !Поправить обратно
            dir_name = os.path.join(self.__data_path, _dir)
            for file in os.listdir(dir_name):
                shutil.copy(os.path.join(dir_name, file), self.__temp_path)
            if (i == count_dirs): # !Удалить
                break

    # Удаляет из midi файлов перкуссионные дорожки и дорожки с unpitched нотами
    def delete_percussion(self):
        files = os.listdir(self.__temp_path)
        for file in files:
            score = converter.parse(os.path.join(self.__temp_path, file))
            new_score = stream.Score()
            for i, part in enumerate(score):
                if (i == 0):
                    continue
                is_part_to_delete = False
                new_part = stream.Part()
                for item in part.recurse():
                    if (isinstance(item, note.Unpitched)
                            or isinstance(item, percussion.PercussionChord)):
                        is_part_to_delete = True
                        break
                    if not ((isinstance(item, note.Note))
                            or (isinstance(item, note.Rest))
                            or (isinstance(item, chord.Chord))
                            or (isinstance(item, stream.Measure))):
                        dur = item.duration.quarterLength
                        tmp_rest = note.Rest()
                        tmp_rest.duration.quarterLength = dur
                        new_part.append(tmp_rest)
                    else:
                        new_part.append(copy.deepcopy(item))
                if (is_part_to_delete):
                    continue
                else:
                    new_score.append(new_part)
            os.remove(os.path.join(self.__temp_path, file))
            new_score.write('midi',
                            fp=os.path.join(self.__temp_path, f"{file}.mid"))

    # Извлекает из очищенных midi файлов ноты, аккорды и паузы, кодирует и сохраняет как .npy файл
    def preprocess(self, n_bars, n_tracks=2, n_steps_per_bar=4):
        step_time = 1 / n_steps_per_bar
        to_npy = []
        files = os.listdir(self.__temp_path)
        for file in files:
            score = converter.parse(os.path.join(self.__temp_path, file))
            score_matrix = [[] for i in range(n_tracks)]
            melody_start_offset = 0
            is_start_find = False
            for i, part in enumerate(score):
                if (i == 0):
                    continue
                if (i == n_tracks + 1):
                    break
                current_time = 0
                for item in part.flatten():
                    if (i == 1 and not is_start_find):
                        if (isinstance(item, note.Note) or isinstance(item, chord.Chord)):
                            is_start_find = True
                        if not (is_start_find):
                            melody_start_offset += item.duration.quarterLength
                    current_time += item.duration.quarterLength
                    if (is_start_find):
                        if ((isinstance(item, note.Note) or isinstance(item, chord.Chord) or isinstance(item, note.Rest))
                                and (current_time >= melody_start_offset)
                                and ((current_time - melody_start_offset) < n_bars)):
                            if (isinstance(item, note.Note)):
                                score_matrix[i - 1] += ([str(item.nameWithOctave)] *
                                                        int(item.duration.quarterLength / step_time))
                            if (isinstance(item, chord.Chord)):
                                score_matrix[i - 1] += ['.'.join(n.nameWithOctave for n in item.pitches)] * int(item.duration.quarterLength / step_time)
                            if (isinstance(item, note.Rest)):
                                score_matrix[i - 1] += ([str(item.name)] *
                                                        int(item.duration.quarterLength / step_time))
            to_npy.append(score_matrix)
        max_len = 0
        all_notes = {'rest'}
        for matrix in to_npy:
            for part in matrix:
                if (max_len < len(part)):
                    all_notes = all_notes.union(set(part))
                    max_len = len(part)
        all_notes.discard('rest')
        all_notes = ['rest'] + list(all_notes)
        for matrix in to_npy:
            for j, part in enumerate(matrix):
                part += ['rest'] * (max_len - len(part))
                matrix[j] = [part.index(n) for n in part]
        to_npy = np.array(to_npy)
        np.save(os.path.join(self.__preprocessed_data_path, "dataset"), to_npy)

    # Удаляет все файлы во временной директории
    def remove_files(self):
        files = os.listdir(self.__temp_path)
        for file in files:
            os.remove(os.path.join(self.__temp_path, file))