import datetime
import logging
import os

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import uuid
from models import *
from cache import RedisHelper
from database import MySqlHelper
from clustering.kmeans import get_nodes_links
import time
from apscheduler.schedulers.background import BackgroundScheduler
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder='../frontend/dist', static_folder='../frontend/dist/static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
mysql = MySqlHelper(app)
# sql = SqlHandler('root', 'xld123456XLD', '192.168.2.174', 'hhctest')
db = mysql.get_db()

redis = RedisHelper()
redis.reset_redis()

sched = BackgroundScheduler(timezone='MST')
sched.add_job(redis.commit_to_sql, 'interval', id='3_second_job', seconds=5, args=[db])
sched.start()


@app.route('/api/list/dynamic', methods=['POST'])
def msg_load_dynamic():
    post_data = request.get_json()
    print('msg_load_dynamic', post_data)
    # todo 效率太低！！！！！！！！！！！！！！
    user_src_list = db.session.query(M_UtS).filter(M_UtS.uid == post_data['token']).all()
    src_info_list = []
    for src_item in user_src_list:
        srcInfo = db.session.query(M_Info).filter(M_Info.sid == src_item.sid).all()
        src_info_list.extend(srcInfo)
    limit = 25
    start = post_data['count'] - 1
    end = start + limit
    info_list = src_info_list[start:end]
    response = {
        'code': 20000,
        'body': {
            'content': [],
            'totalPages': '',
        }
    }
    for data in info_list:
        srcName = db.session.query(M_Src).filter(M_Src.sid == data.sid).one().sname
        temp = {
            "postSrc": '{}'.format(srcName),
            "postTitle": '{}'.format(data.ititle),
            "postLink": '{}'.format(data.ilink),
            'postContent': '{}'.format(data.isummer),
            'postUpdated': '{}'.format(data.iupdated),
            'postLikes': '{}'.format(data.ilikes),
            'postID': '{}'.format(data.iid),
        }
        response['body']['content'].append(temp)
    response['body']['totalPages'] = '{}'.format(len(src_info_list))
    return jsonify(response)


@app.route('/api/list/likes', methods=['POST'])
def postLikes():
    post_data = request.get_json()
    print('postLikes', post_data)

    redis.add_likes(post_data['ID'])

    response = {
        'code': 20000,
        'body': {
            'data': 'success'
        }
    }
    return jsonify(response)


@app.route('/api/list/export', methods=['POST'])
def msg_load_export():
    post_data = request.get_json()
    print('msg_load_export', post_data)
    limit = 25
    start = post_data['count'] - 1
    end = start + limit
    data_list = db.session.query(M_Info).slice(start, end).all()
    response = {
        'code': 20000,
        'body': {
            'content': [],
            'totalPages': '',
        }
    }
    for data in data_list:
        temp = {
            "postTitle": '{}'.format(data.ititle),
            'postContent': '{}'.format(data.isummer)
        }
        response['body']['content'].append(temp)
    response['body']['totalPages'] = '{}'.format(len(data_list))
    return jsonify(response)


@app.route('/api/list/anls', methods=['POST'])
def msg_load_anls():
    post_data = request.get_json()
    print('msg_load_anls', post_data)
    limit = 25
    start = post_data['count'] - 1
    end = start + limit
    data_list = db.session.query(M_Info).slice(start, end).all()
    response = {
        'code': 20000,
        'body': {
            'content': [],
            'totalPages': '',
        }
    }
    for data in data_list:
        temp = {
            "postTitle": '{}'.format(data.ititle),
            'postContent': '{}'.format(data.isummer)
        }
        response['body']['content'].append(temp)
    response['body']['totalPages'] = '100'
    return jsonify(response)


@app.route('/api/list/getGraph', methods=['POST'])
def getGraph():
    post_data = request.get_json()
    print('getGraph', post_data)
    # TODO
    response = {
        'code': 20000,
        'data': {
            'nodes': [],
            'links': [],
        }
    }
    nodes, links = get_nodes_links('./clustering/stopwords-master/baidu_stopwords.txt')
    print(len(nodes), len(links))
    for item in nodes:
        tmp = {'id': item['id'], 'group': item['group']}
        response['data']['nodes'].append(tmp)
    for item in links:
        tmp = {'source': item['source'], 'target': item['target'], 'value': item['value']}
        response['data']['links'].append(tmp)
    return jsonify(response)


@app.route('/api/list/srcList', methods=['POST'])
def msg_load_src():
    post_data = request.get_json()
    print('msg_load_src', post_data)
    # todo 效率太低！！！！！！！！！！！！！！
    src_list = db.session.query(M_Src).all()
    limit = 25
    start = post_data['count'] - 1
    end = start + limit
    src_list_splice = src_list[start:end]
    response = {
        'code': 20000,
        'body': {
            'content': [],
            'totalPages': '',
        }
    }
    for data in src_list_splice:
        temp = {
            "sid": '{}'.format(data.sid),
            'sname': '{}'.format(data.sname),
            'surl': '{}'.format(data.surl),
            'ssummer': '{}'.format(data.ssummer)
        }
        response['body']['content'].append(temp)
    response['body']['totalPages'] = '{}'.format(int(len(src_list_splice) / limit) + 1)
    return jsonify(response)


@app.route('/api/user/login', methods=['POST'])
def user_login():
    print('user_login')
    post_data = request.get_json()
    data_list = db.session.query(M_User).filter(M_User.uname == post_data['username']).all()
    new_log = M_Loginlog(id=str(uuid.uuid1()),
                         ip=str(request.remote_addr),
                         name=post_data['username'],
                         password=post_data['password'],
                         date=datetime.datetime.now().strftime('%Y_%m_%d-%H:%M'))
    db.session.add(new_log)
    db.session.commit()
    response = {
        'code': 0,
        'data': {
            'token': ''
        }
    }
    if len(data_list) == 0:  # 注册
        temp_uid = uuid.uuid1()
        new_user = M_User(uid=temp_uid,
                          uname=post_data['username'],
                          upassword=post_data['password'])
        db.session.add(new_user)
        db.session.commit()
        response['code'] = 20000
        response['data']['token'] = str(temp_uid)
    else:  # 已存在，验证密码
        if post_data['password'] == data_list[0].upassword:
            response['code'] = 20000
            response['data']['token'] = str(data_list[0].uid)
        else:
            response['code'] = 20001
            response['data']['token'] = str(data_list[0].uid)
    print(response)
    return jsonify(response)


@app.route('/api/user/info', methods=['GET'])
def user_info():
    print('user_info', request.values.get('token'))
    post_data = request.values.get('token')
    data_list = db.session.query(M_User).filter(M_User.uid == post_data).all()
    response = {
        'code': 20000,
        'data': {
            'token': '{}'.format(data_list[0].uid),
            'name': '{}'.format(data_list[0].uname),
        }
    }
    return jsonify(response)


@app.route('/api/user/logout', methods=['POST'])
def user_logout():
    print('user_logout')
    post_data = request.get_json()
    print(post_data)
    response = {
        'code': 20000,
        'data': 'success',
    }
    return jsonify(response)


@app.route('/api/userSidebar/getSrc', methods=['POST'])
def user_getSrc():
    post_data = request.get_json()
    print('user_getSrc', post_data['token'])
    data_list = db.session.query(M_UtS).filter(M_UtS.uid == post_data['token']).all()
    src_list = []
    for item in data_list:
        src = db.session.query(M_Src).filter(M_Src.sid == item.sid).all()
        src_list.extend(src)
    response = {
        'code': 20000,
        'body': {
            'content': [],
        },
    }
    for data in src_list:
        temp = {
            "src_id": '{}'.format(data.sid),
            'src_name': '{}'.format(data.sname)
        }
        response['body']['content'].append(temp)
    return jsonify(response)


@app.route('/api/userSidebar/deleteOne', methods=['POST'])
def user_src_deleteOne():
    post_data = request.get_json()
    print('user_src_deleteOne', post_data['row']['src_name'])
    one_uts = db.session.query(M_UtS).filter(M_UtS.uid == post_data['token'],
                                     M_UtS.sid == post_data['row']['src_id']).one()
    db.session.delete(one_uts)
    db.session.commit()
    response = {
        'code': 20000,
        'data': 'success',
    }
    return jsonify(response)


@app.route('/api/userSidebar/addOne', methods=['POST'])
def user_src_addOne():
    post_data = request.get_json()
    print('user_src_addOne', post_data['sid'])
    data_list = db.session.query(M_UtS).filter(M_UtS.uid == post_data['token'],
                                       M_UtS.sid == post_data['sid']).all()
    if len(data_list) == 0:
        new_item = M_UtS(id=uuid.uuid1(),
                         uid=post_data['token'],
                         sid=post_data['sid'])
        db.session.add(new_item)
        db.session.commit()
    response = {
        'code': 20000,
        'data': 'success',
    }
    return jsonify(response)


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    if app.debug:
        return requests.get('http://localhost:8080/{}'.format(path)).text
    return render_template("index.html")


if __name__ == '__main__':
    handler = logging.FileHandler('./flask.log')
    app.logger.addHandler(handler)
    app.run()
