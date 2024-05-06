import os
import shutil
import copy

from music21 import converter, note, chord, percussion, stream
import numpy as np


class DataLoader:
    def __init__(self, data_filename, _format='npy'):
        self.__data_path = os.path.join(f"./datasets/RefactorData/data/{data_filename}.{_format}")
        self.__tags_path = os.path.join(f"./datasets/RefactorData/data/{data_filename}.txt")
        self.__format = _format

    def get_dataset(self):
        try:
            with open(self.__tags_path, 'r') as tags_file:
                tags = tags_file.read().split()

            if (self.__format == 'npy'):
                return np.load(self.__data_path), tags
            else:
                print('Указанный формат не поддерживается')
                return None, None
        except:
            print("Указанный файл не найден")
            return None, None


class DataConverter:
    def __init__(self, directory):
        self.__data_path = os.path.join(f"./datasets/{directory}")
        self.__temp_path = os.path.join("./datasets/RefactorData/temp")
        self.__preprocessed_data_path = os.path.join("./datasets/RefactorData/data")

    # Извлекает все midi файлы во временную директорию
    def extract_files(self):
        directories = os.listdir(self.__data_path)
        for i, _dir in enumerate(directories):
            dir_name = os.path.join(self.__data_path, _dir)
            for j, file in enumerate(os.listdir(dir_name)):
                shutil.copy(os.path.join(dir_name, file), self.__temp_path)
        with open("datasets/to_delete.txt") as file:
            for line in file:
                strip_line = line.strip('\n')
                os.remove(os.path.join(f"./datasets/RefactorData/temp/{strip_line}"))
        print("log_message: files was extracted\n")

    # Удаляет из midi файлов перкуссионные дорожки и дорожки с unpitched нотами
    def delete_percussion(self):
        files = os.listdir(self.__temp_path)
        print("log_message: start deleting percussion\nprogress:")
        for i, file in enumerate(files): # Убрать енам
            print(file)
            if (i > 1000): # Убрать
                break
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
                            fp=os.path.join(self.__temp_path, file))
        print("log_message: percussion was deleted\n")

    # Извлекает из очищенных midi файлов ноты, аккорды и паузы, кодирует и сохраняет как .npy файл

    def preprocess(self, n_bars, n_tracks=2, n_steps_per_bar=4, output_length=200):
        step_time = 1 / n_steps_per_bar
        to_npy_tmp = []
        files = os.listdir(self.__temp_path)
        for k, file in enumerate(files): # убрать енам
            print(f"log_message: current track is {file}")
            if (k > 200): # убрать
                break
            score = converter.parse(os.path.join(self.__temp_path, file))
            score_matrix = []
            offset = self.search_offset(score, n_bars, step_time)

            notes = []
            count_notes_without_rest = []
            priority = []
            for i, part in enumerate(score):
                if (i == 0):
                    continue
                notes.append(self.extract_part(part, offset, n_bars, step_time))
                count_notes_without_rest.append(len([note for note in notes[len(notes) - 1] if (note != 'rest')]))

            if (n_tracks >= len(notes)):
                for i in range(len(notes)):
                    score_matrix.append(notes[i])
                while (len(score_matrix) != n_tracks):
                    score_matrix.append(['rest'])
            else:
                for i in range(n_tracks):
                    max_count_notes_ix = count_notes_without_rest.index(max(count_notes_without_rest))
                    score_matrix.append(notes[max_count_notes_ix])
                    count_notes_without_rest[max_count_notes_ix] = -1
            to_npy_tmp.append(score_matrix)

        # Извлекаем output_length аудио с наибольшим суммарным числом нот
        count_notes_without_rest_in_score = [0] * 1000
        to_npy = []
        for i, track in enumerate(to_npy_tmp):
            count_notes_without_rest_in_score[i] += len([note for note in track[len(track) - 1] if (note != 'rest')])
        for i in range(output_length):
            max_count_notes_ix = count_notes_without_rest_in_score.index(max(count_notes_without_rest_in_score))
            to_npy.append(to_npy_tmp[max_count_notes_ix])
            count_notes_without_rest_in_score[max_count_notes_ix] = -1

        max_len = 0
        all_notes = {'rest'}
        for matrix in to_npy:
            for part in matrix:
                all_notes = all_notes.union(set(part))
                if (max_len < len(part)):
                    max_len = len(part)
        all_notes.discard('rest')
        all_notes = ['rest'] + list(all_notes)

        # Добавляем в конец паузу, чтобы итоговые матрицы имели подходящую размерность
        count_to_add = 1 # Кастомно

        for matrix in to_npy:
            for j, part in enumerate(matrix):
                part += ['rest'] * (max_len - len(part) + count_to_add)
                matrix[j] = [all_notes.index(n) for n in part]
        to_npy = np.array(to_npy)

        tmp = []
        for i in range(output_length):
            tmp.append(to_npy[i].transpose())
        to_npy = np.array(tmp)

        print("log_message: data was preprocessed")
        print(f"log_message: total data shape is {to_npy.shape}")
        print(f"log_message: total note count is {len(all_notes)}\n")
        np.save(os.path.join(self.__preprocessed_data_path, "dataset"), to_npy)
        with open("datasets/RefactorData/data/dataset.txt", "w") as tags_file:
            tags_file.write(' '.join(all_notes))



    # Вспомогательная. Возвращает извлеченные из партии ноты
    def extract_part(self, part, offset, n_bars, step_time):
        current_time = 0
        result = []
        for item in part.flatten():
            current_time += item.duration.quarterLength
            if ((isinstance(item, note.Note) or isinstance(item, chord.Chord) or isinstance(item, note.Rest))
                    and (current_time >= offset)
                    and ((current_time - offset) < n_bars)):
                if (isinstance(item, note.Note)):
                    result += ([str(item.nameWithOctave)] *
                               int(item.duration.quarterLength / step_time))
                # if (isinstance(item, chord.Chord)): # Из-за слишком большого числа параметров было решено вырезать аккорды
                #     result += (['.'.join(n.nameWithOctave for n in item.pitches)] *
                #                int(item.duration.quarterLength / step_time))
                if (isinstance(item, note.Rest)):
                    result += ([str(item.name)] *
                               int(item.duration.quarterLength / step_time))
        return result

    # Вспомогательная. Возвращает offset, дающий наибольшее число нот
    def search_offset(self, score, n_bars, step_time):
        count_notes = []
        offsets = []
        for i, part in enumerate(score):
            if (i == 0):
                continue
            offset = 0
            current_time = 0
            for item in part.flatten():
                if (isinstance(item, note.Note)): # if (isinstance(item, note.Note) or isinstance(item, chord.Chord)):
                    offset = current_time
                    break
                current_time += item.duration.quarterLength
            count_notes.append(len([note for note in self.extract_part(part, offset, n_bars, step_time) if note != 'rest']))
            offsets.append(offset)
        max_count_notes_ix = count_notes.index(max(count_notes)) if (len(count_notes) > 0) else -1
        return offsets[max_count_notes_ix] if (max_count_notes_ix != -1) else 0

    # Удаляет все файлы во временной директории
    def remove_files(self):
        files = os.listdir(self.__temp_path)
        for file in files:
            os.remove(os.path.join(self.__temp_path, file))
        print("log_message: temp directory was cleared")


# class DataRestorer:
#     def __init__(self, tags):
#         self.__tags = tags.copy()
#         self.__output_dir = os.path.join(f"./output")
#
#     def restore(self, data, filename):
#         uncoding_data = []
#         for track in data:
#             tmp_track = []
#             for n in track:
#                 tmp_track.append(self.__tags[n])
#             uncoding_data.append(tmp_track)
#
#         notes = []
#         durations = []
#         for track in uncoding_data:
#             tmp_note_track = []
#             tmp_dur_track = []
#             for i, n in enumerate(track):
#                 if (i == 0):
#                     prev_n = n
#                     cur_dur = 0.25
#                 if ((prev_n == n) and (i != 0)):
#                     cur_dur += 0.25
#                 if ((prev_n != n) and (i != 0)):
#                     tmp_note_track.append(prev_n)
#                     tmp_dur_track.append(cur_dur)
#                     prev_n = n
#                     cur_dur = 0.25
#             tmp_note_track.append(prev_n)
#             tmp_dur_track.append(cur_dur)
#             notes.append(tmp_note_track)
#             durations.append(tmp_dur_track)
#
#         new_score = stream.Score()
#         for i in range(len(notes)):
#             new_track = stream.Stream()
#             for n, dur in zip(notes[i], durations[i]):
#                 if (n == 'rest'):
#                     _note = note.Rest()
#                 elif (len(n.split('.')) > 1):
#                     _note = chord.Chord(n.split('.'))
#                 else:
#                     _note = note.Note(n)
#                 _note.duration.quarterLength = dur
#                 new_track.append(_note)
#             new_score.append(new_track)
#         new_score.write('midi', fp=os.path.join(self.__output_dir, f"{filename}.mid"))




