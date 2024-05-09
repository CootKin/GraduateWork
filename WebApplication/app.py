import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from flask import Flask, render_template, url_for, request, redirect, flash, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.sql import text
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import model.model as network
import numpy as np
import music21
import json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal

app = Flask(__name__, static_folder="static")

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'

db = SQLAlchemy(app)

class Tariff(db.Model):
    tariff_id = db.Column(db.Integer, primary_key=True) #bigint pk
    count_tokens_per_minute = db.Column(db.Integer, nullable=False) #bigint
    generation_cost = db.Column(db.Integer, nullable=False) #bigint

    def __repr__(self):
        return f"<tariff {self.tariff_id}>"

class User(db.Model):
    user_id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(50), nullable=True)
    user_login = db.Column(db.String(50), unique=True, nullable=False)
    user_password = db.Column(db.String(100), nullable=False)
    tariff_id = db.Column(db.Integer, db.ForeignKey('tariff.tariff_id'), nullable=False)
    creation_date = db.Column(db.DateTime, default=datetime.now())

    def __repr__(self):
        return f"<user {self.user_id}>"

class Files(db.Model):
    file_id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    generation_date = db.Column(db.DateTime, default=datetime.now())
    generation_params = db.Column(JSON)
    file_size = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return f"<files {self.file_id}>"

class CurrentUserInfo:
    def __init__(self, user_id=None, user_name=None, user_login=None, user_current_tokens=None, generation_cost=None, creation_date=None, count_generated_files=None, count_tokens_per_minute=None):
        self.user_id = user_id
        self.user_name = user_name
        self.user_login = user_login
        self.user_current_tokens = user_current_tokens
        self.generation_cost = generation_cost
        self.creation_date = creation_date
        self.count_generated_files = count_generated_files
        self.count_tokens_per_minute = count_tokens_per_minute

    def set_new_user_info(self, user_id, user_name, user_login, user_current_tokens, generation_cost, creation_date, count_generated_files, count_tokens_per_minute):
        self.user_id = user_id
        self.user_name = user_name
        self.user_login = user_login
        self.user_current_tokens = user_current_tokens
        self.generation_cost = generation_cost
        self.creation_date = creation_date
        self.count_generated_files = count_generated_files
        self.count_tokens_per_minute = count_tokens_per_minute

    def update_current_tokens(self, current_tokens):
        self.user_current_tokens = current_tokens

    def increment_count_generated_files(self):
        self.count_generated_files += 1

    def update_username(self, username):
        self.user_name = username

class CurrentFileInfo:
    def __init__(self, file_id=None, file_name=None, user_id=None, generation_date=None, generation_params=None, file_size=None):
        self.file_id = file_id
        self.file_name = file_name
        self.user_id = user_id
        self.generation_date = generation_date
        self.generation_params = generation_params
        self.file_size = file_size

    def set_new_file_info(self, file_id, file_name, user_id, generation_date, generation_params, file_size):
        self.file_id = file_id
        self.file_name = file_name
        self.user_id = user_id
        self.generation_date = generation_date
        self.generation_params = generation_params
        self.file_size = file_size

    def set_new_file(self, file_id, file_name, generation_date, generation_params, file_size):
        self.file_id = file_id
        self.file_name = file_name
        self.generation_date = generation_date
        self.generation_params = generation_params
        self.file_size = file_size

class HelpClass:
    def __init__(self):
        self.generated = False
        self.download_enabled = False
        self.freeze_noise = [False] * 4

    def set_generated(self, value):
        self.generated = value

    def set_download_enabled(self, value):
        self.download_enabled = value

    def set_freeze_noise(self, value, idx):
        self.freeze_noise[idx] = value


### Загрузка модели ##
with open('./model/tags.txt', 'r') as tags_file:
    tags = tags_file.read().split()
initializer = RandomNormal(mean=0.0, stddev=0.02)

model = network.MuseGAN(
    discriminator=network.discriminator_initialize(
        initializer, 9, 16, 75, 4
    ),
    generator=network.generator_initialize(
        initializer, 32, 4, 9, 16, 75
    ),
    noise_length=32,
    count_tracks=4,
    discriminator_steps=5,
    gradient_penalty_weight=10
)
model.compile(
    discriminator_optimizer=Adam(0.001,0.5,0.9),
    generator_optimizer=Adam(0.001,0.5,0.9)
)
model.load_weights("./checkpoint/checkpoint_1000/checkpoint.ckpt")

def generate_music(model, tags, file_path):
    prev_noise = [None] * 4
    if (current_file.file_id != None):
        prev_noise = [np.array(noise) for noise in current_file.generation_params]

    notes_set = set()
    while len(notes_set) < 5:
        if (prev_noise[0] is not None and not help_class.freeze_noise[0]):
            chords_noise = np.random.normal(size=(1, 32))
        else:
            if prev_noise[0] is None:
                chords_noise = np.random.normal(size=(1, 32))
            else:
                chords_noise = prev_noise[0]

        if (prev_noise[0] is not None and not help_class.freeze_noise[1]):
            style_noise = np.random.normal(size=(1, 32))
        else:
            if prev_noise[0] is None:
                style_noise = np.random.normal(size=(1, 32))
            else:
                style_noise = prev_noise[1]

        if (prev_noise[0] is not None and not help_class.freeze_noise[2]):
            melody_noise = np.random.normal(size=(1, 4, 32))
        else:
            if prev_noise[0] is None:
                melody_noise = np.random.normal(size=(1, 4, 32))
            else:
                melody_noise = prev_noise[2]

        if (prev_noise[0] is not None and not help_class.freeze_noise[3]):
            groove_noise = np.random.normal(size=(1, 4, 32))
        else:
            if prev_noise[0] is None:
                groove_noise = np.random.normal(size=(1, 4, 32))
            else:
                groove_noise = prev_noise[3]

        noise_vectors = [
            chords_noise,
            style_noise,
            melody_noise,
            groove_noise,
        ]

        generator_output = model.generator(noise_vectors).numpy()
        max_pitches = np.argmax(generator_output, axis=3)
        generated_notes = max_pitches.reshape([9 * 16, 4])
        notes_set = set()
        for step in generated_notes:
            notes_set = notes_set.union(set(step))

    parts = music21.stream.Score()
    parts.append(music21.tempo.MetronomeMark(number=66))

    for i in range(4):
        current_code_note = int(generated_notes[:, i][0])
        new_stream = music21.stream.Part()
        dur = 0
        for idx, code_note in enumerate(generated_notes[:, i]):
            code_note = int(code_note)
            if (code_note != current_code_note or idx % 4 == 0) and idx > 0:
                if (current_code_note == 0):
                    uncode_note = music21.note.Rest()
                else:
                    uncode_note = music21.note.Note(tags[current_code_note])
                uncode_note.duration = music21.duration.Duration(dur)
                new_stream.append(uncode_note)
                dur = 0
            current_code_note = code_note
            dur = dur + 0.25
        if (current_code_note == 0):
            uncode_note = music21.note.Rest()
        else:
            uncode_note = music21.note.Note(tags[current_code_note])
        uncode_note.duration = music21.duration.Duration(dur)
        new_stream.append(uncode_note)

        parts.append(new_stream)
    parts.write("midi", fp=f"{file_path}")

    file_name = file_path.split('/')
    file_name = file_name[len(file_name) - 1]
    noise_vectors[0] = [list(noise_vectors[0][0])]
    noise_vectors[1] = [list(noise_vectors[1][0])]
    noise_vectors[2] = [[list(vector) for vector in noise_vectors[2][0]]]
    noise_vectors[3] = [[list(vector) for vector in noise_vectors[3][0]]]
    return list(noise_vectors), file_name, datetime.now()



# app.app_context().push()
# db.create_all()
# db.session.add(Tariff(count_tokens_per_minute=5, generation_cost=1))
# db.session.add(Tariff(count_tokens_per_minute=10, generation_cost=5))
# db.session.commit()


### Представление данных пользователя ###
current_user = CurrentUserInfo()
current_file = CurrentFileInfo()
help_class = HelpClass()

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        try:
            password = request.form["password"]
            login = request.form["login"]

            user_is_exists = db.session.execute(text(f"SELECT COUNT(*) FROM user WHERE user_login == :login"), {"login": login}).all()
            if (user_is_exists[0][0] == 0):
                return render_template("auth.html", message="user_not_exists")
            else:
                user_password = db.session.execute(text(f"SELECT user_password FROM user WHERE user_login == :login"), {"login": login}).all()
                if check_password_hash(user_password[0][0], password):
                    user_id = db.session.execute(text(f"SELECT user_id FROM user WHERE user_login == :login"), {"login": login}).all()
                    user_name = db.session.execute(text(f"SELECT user_name FROM user WHERE user_login == :login"), {"login": login}).all()
                    generation_cost = db.session.execute(text(f"SELECT generation_cost FROM user JOIN tariff USING(tariff_id) WHERE user_login == :login"), {"login": login}).all()
                    creation_date = db.session.execute(text(f"SELECT creation_date FROM user WHERE user_login == :login"), {"login": login}).all()
                    count_generated_files = db.session.execute(text(f"SELECT COUNT(*) FROM user JOIN files USING(user_id) WHERE user_login == :login"), {"login": login}).all()
                    count_tokens_per_minute = db.session.execute(text(f"SELECT count_tokens_per_minute FROM user JOIN tariff USING(tariff_id) WHERE user_login == :login"), {"login": login}).all()
                    current_user.set_new_user_info(user_id[0][0], user_name[0][0], login, 0, generation_cost[0][0], creation_date[0][0], count_generated_files[0][0], count_tokens_per_minute[0][0])
                    current_file.set_new_file_info(None, None, user_id, None, None, None)
                    return redirect("/main")
                else:
                    return render_template("auth.html", message="invalid_password", login=login)
        except:
            return render_template("auth.html", message="invalid_data")
    else:
        return render_template("auth.html")

@app.route('/register', methods=['POST', 'GET'])
def register():
    if request.method == 'POST':
        try:
            password_hash = generate_password_hash(request.form["password"])
            login = request.form["login"]
            username = request.form["username"]

            user_is_exists = db.session.execute(text(f"SELECT COUNT(*) FROM user WHERE user_login == :login"), {"login": login}).all()
            if (user_is_exists[0][0] == 0):
                new_user = User(user_name=username, user_login=login, user_password=password_hash, tariff_id=1)
                db.session.add(new_user)
                db.session.commit()
                os.makedirs(os.path.join('./static/files', login))
                return redirect("/")
            else:
                return render_template("register.html", message="user_exist")
        except:
            return render_template("register.html", message="invalid_data")
    else:
        return render_template("register.html")

@app.route('/main', methods=['POST', 'GET'])
def main():
    date_difference = datetime.now() - datetime.strptime(current_user.creation_date, '%Y-%m-%d %H:%M:%S.%f')
    count_minutes = int(date_difference.days*24*60 + date_difference.seconds/60)
    current_user_tokens = count_minutes * current_user.count_tokens_per_minute - current_user.count_generated_files * current_user.generation_cost
    current_user.update_current_tokens(current_user_tokens)

    if request.method == 'POST':
        try:
            try_get_req = request.form["is_download"]
            user_files_path = os.path.join("./static/files", current_user.user_login)
            return send_file(os.path.join(user_files_path, current_file.file_name), as_attachment=True)
        except:
            print("Нажата кнопка генерации")

        try:
            if (current_user_tokens < current_user.generation_cost):
                return redirect("/main")

            freeze_noise0 = "checkbox4" in request.form
            freeze_noise1 = "checkbox1" in request.form
            freeze_noise2 = "checkbox2" in request.form
            freeze_noise3 = "checkbox3" in request.form
            help_class.set_freeze_noise(freeze_noise0, 0)
            help_class.set_freeze_noise(freeze_noise1, 1)
            help_class.set_freeze_noise(freeze_noise2, 2)
            help_class.set_freeze_noise(freeze_noise3, 3)

            current_user.increment_count_generated_files()
            date_difference = datetime.now() - datetime.strptime(current_user.creation_date, '%Y-%m-%d %H:%M:%S.%f')
            count_minutes = int(date_difference.days * 24 * 60 + date_difference.seconds / 60)
            current_user_tokens = count_minutes * current_user.count_tokens_per_minute - current_user.count_generated_files * current_user.generation_cost
            current_user.update_current_tokens(current_user_tokens)

            return redirect("/generation_process")
        except:
            return redirect("/main")
    else:
        if (help_class.download_enabled):
            return render_template("generate.html", tokens=current_user.user_current_tokens, message="download_active",
                                   checkbox1=str(help_class.freeze_noise[1]),
                                   checkbox2=str(help_class.freeze_noise[2]),
                                   checkbox3=str(help_class.freeze_noise[3]),
                                   checkbox4=str(help_class.freeze_noise[0])
                                   )
        else:
            return render_template("generate.html", tokens=current_user.user_current_tokens,
                                   checkbox1=str(help_class.freeze_noise[1]),
                                   checkbox2=str(help_class.freeze_noise[2]),
                                   checkbox3=str(help_class.freeze_noise[3]),
                                   checkbox4=str(help_class.freeze_noise[0])
                                   )

@app.route('/generation_process', methods=['POST', 'GET'])
def generation_process():
    if request.method == 'POST':
        user_files_path = f"./static/files/{current_user.user_login}"
        count_files = len(os.listdir(user_files_path))
        noise_vectors, file_name, generation_date = generate_music(model, tags, f"{user_files_path}/{count_files}.midi")
        noise_vectors_json = json.dumps(noise_vectors, indent=4, ensure_ascii=False)
        file_size = os.path.getsize(f"{user_files_path}/{count_files}.midi")
        file_size = f"{file_size} байт" if (file_size / 1024 < 1) else f"{file_size / 1024} Кбайт"

        new_file = Files(file_name=file_name, user_id=current_user.user_id, generation_date=generation_date, generation_params=noise_vectors_json, file_size=file_size)
        db.session.add(new_file)
        db.session.commit()

        file_id = db.session.execute(text(f"SELECT file_id FROM files WHERE file_name == :file_name"), {"file_name": file_name}).all()
        current_file.set_new_file(file_id[0][0], file_name, generation_date, noise_vectors, file_size)

        help_class.set_download_enabled(True)
        return redirect('/main')
    else:
        return render_template("generation_process.html", tokens=current_user.user_current_tokens,
                               checkbox1=str(help_class.freeze_noise[1]),
                               checkbox2=str(help_class.freeze_noise[2]),
                               checkbox3=str(help_class.freeze_noise[3]),
                               checkbox4=str(help_class.freeze_noise[0])
                               )

@app.route('/lk_history', methods=['GET', 'POST'])
def lk_history():
    if request.method == 'POST':
        try:
            filename = request.form["filename"]
            user_files_path = os.path.join("./static/files", current_user.user_login)
            return send_file(os.path.join(user_files_path, filename), as_attachment=True)
        except:
            return redirect("/")
    else:
        files = Files.query.order_by(Files.generation_date.desc()).where(Files.user_id == current_user.user_id).all()
        return render_template("lk_history.html", tokens=current_user.user_current_tokens, files=files)

@app.route('/lk_settings', methods=['POST', 'GET'])
def lk_settings():
    if request.method == 'POST':
        try:
            username = request.form["username"]
            password = request.form["password"]

            if (username != "" and username is not None) or (password != "" and password is not None):
                user = User.query.get(current_user.user_id)
                if (username != "" and username is not None):
                    current_user.update_username(username)
                    user.user_name = username
                if (password != "" and password is not None):
                    password_hash = generate_password_hash(password)
                    user.user_password = password_hash
                db.session.commit()
            return redirect("/lk_settings")
        except:
            return redirect("/lk_settings")
    else:
        user_name = f", {current_user.user_name}" if (current_user.user_name != "" and current_user.user_name is not None) else ""
        return render_template("lk_settings.html", tokens=current_user.user_current_tokens, user_name=user_name)



if __name__ == "__main__":
    app.run(host="localhost", port="5000", debug=False)