import asyncio
import datetime
import re
import uuid
import aiohttp
import feedparser
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from flask_sqlalchemy import SQLAlchemy
from backend.models import *
import time
from datetime import datetime


def transTimeFormat(string):
    str_list = string.split()
    if len(str_list) > 1:
        str_list.pop(-1)
        if not str_list[0][0].isdigit():
            str_list.pop(0)
    str_list = ' '.join(str_list)

    strs = str_list
    tex = [
        '%d %b %Y %H:%M:%S',  # Sun, 05 May 2019 00:00:00
        '%Y-%m-%d %H:%M:%S',  # 2021-04-16 16:40:32
        '%Y-%m-%dT%H:%M:%SZ',  # 2021-03-26T04:43:00Z
    ]
    for j in range(len(tex)):
        try:
            d = datetime.strptime(strs.strip(), tex[j])
            stamp = datetime.timestamp(d)
            time_local = time.localtime(stamp)
            dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
            return dt
        except ValueError:
            continue
    return string


class MySession:
    def __init__(self, sess):
        self.session = sess


class MySqlHelper:
    def __init__(self, app=None):
        user = 'root'
        passwd = 'password'
        # passwd = 'xld123456XLD'
        host = 'localhost'
        # host = '192.168.2.174'
        name = 'hhctest'
        SQLALCHEMY_DATABASE_URI = "mysql+pymysql://{}:{}@{}:3306/{}?charset=utf8mb4".format(user, passwd, host, name)
        if app is not None:
            app.config["SQLALCHEMY_DATABASE_URI"] = SQLALCHEMY_DATABASE_URI
            app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
            self._db = SQLAlchemy(app)
        else:
            engine = create_engine(SQLALCHEMY_DATABASE_URI)
            Session = sessionmaker(bind=engine)
            session = Session()
            self._db = MySession(session)

    def get_db(self):
        return self._db

    def clear_all(self, arg):
        if arg == 'info':
            self._db.session.query(M_Info).delete()
        elif arg == 'src':
            self._db.session.query(M_Src).delete()
        self._db.session.commit()
        print('delete ok')

    def src_addSrc(self, src_list):
        for src in src_list:
            if len(self._db.session.query(M_Src).filter(M_Src.sname == src[0]).all()) != 0:
                print('insert fail! exists: {}'.format(src[0]))
            else:
                new_src = M_Src(sid=str(uuid.uuid1()),
                                sname=src[0],
                                surl=src[1],
                                supdated='2020')
                self._db.session.add(new_src)
                print('insert finished! exists: {}'.format(src[0]))
        # 提交即保存到数据库
        self._db.session.commit()

    def info_reflashAll(self, src_name_list=None, all_force=False):
        update_num = {}  # (src.stitle - cnt) 每个源更新的条目数量
        src_item_list = []  # 根据src更新对应info

        if src_name_list is not None:
            for src_name in src_name_list:
                src_item_list.append(self._db.session.query(M_Src).filter(M_Src.sname == src_name[0]).one())
        else:
            src_item_list = self._db.session.query(M_Src).all()
        print('({}) src loaded...'.format(len(src_item_list)))

        async def handle_tasks(task_id, work_queue, RSS_dicts):
            while not work_queue.empty():
                src_item = await work_queue.get()
                print('Requesting for feedparser: {}'.format(src_item.sname))
                async with aiohttp.ClientSession() as session:
                    async with session.get(src_item.surl) as response:
                        body = await response.text()
                        RSS_dicts[src_item.sid] = feedparser.parse(body)
                print('Requesting ok {}'.format(src_item.sname))

        RSS_dicts = {}
        q = asyncio.Queue()
        [q.put_nowait(src_item) for src_item in src_item_list]
        loop = asyncio.get_event_loop()
        tasks = [handle_tasks(task_id, q, RSS_dicts, ) for task_id in range(10)]
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

        for src_item in src_item_list:
            # 获取当前更新时间
            RSS_dict = RSS_dicts[src_item.sid]
            if 'updated' in RSS_dict['feed']:
                updated_now = RSS_dict['feed']['updated']
            else:
                updated_now = 'no val'
            updated_local = src_item.supdated
            if (updated_now != 'no val' and updated_local != updated_now) or all_force:
                # 刷新src中的更新时间
                src_item.supdated = updated_now
                # 添加info中的条目
                add_cnt = 0
                for entry in RSS_dict['entries']:
                    ITitle = entry['title']
                    ILink = entry['link']
                    temp_s = entry['summary']
                    dr = re.compile(r'<[^>]+>', re.S)
                    temp_s = dr.sub('', temp_s)
                    if len(temp_s) > 190:
                        temp_s = temp_s[0:190]
                    ISummer = temp_s
                    if 'published' in entry:
                        IUpdated = transTimeFormat(str(entry['published']))
                    else:
                        IUpdated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    if len(self._db.session.query(M_Info).filter(M_Info.ititle == ITitle).all()) != 0:
                        pass
                        # print('update fail! is exists: {}'.format(ITitle))
                    else:
                        new_info = M_Info(iid=str(uuid.uuid1()),
                                          sid=src_item.sid,
                                          ititle=ITitle,
                                          ilink=ILink,
                                          isummer=ISummer,
                                          iupdated=IUpdated,
                                          ilikes=0)
                        self._db.session.add(new_info)
                        add_cnt += 1
                        # print('update finished! add exists: {}'.format(ITitle))
                update_num[src_item.sname] = add_cnt
                print('update finished!\tname: {}\tcnt: {}'.format(src_item.sname, add_cnt))
            else:
                print('updated fail!\tname: {}\tcnt: {}'.format(src_item.sname, 0))
        # 提交即保存到数据库
        self._db.session.commit()
        print('check finished!')

        return update_num

    def info_getAll(self, flag):
        all_info = self._db.session.query(M_Info).all()
        if flag == 'title':
            ans = [x.ititle for x in all_info]
        elif flag == 'summer':
            ans = [x.isummer for x in all_info]
        else:
            ans = [x.iid for x in all_info]
        return ans

    def uts_removeSrc(self, uid, sid):
        pass

    def uts_addSrc(self):
        pass

    def test(self):
        self._db.session.query(M_Info).order_by(M_Info.iupdated.desc()).all()
        pass


def main():
    sql = MySqlHelper()
    # sql = SqlHandler('root', 'xld123456XLD', '192.168.2.174', 'hhctest')
    src_list = [
        # ['BBC_News', 'http://feeds.bbci.co.uk/news/rss.xml'],
        # ['Engadget', 'http://www.engadget.com/rss.xml'],
        # ['Entrepreneur', 'http://feeds.feedburner.com/entrepreneur/latest'],
        # ['Yanko_Design', 'http://www.yankodesign.com/feed/'],
        # ['TEDTalks(video)', 'https://www.ted.com/feeds/talks.rss'],
        # ['FAIL_Blog', 'http://feeds.feedburner.com/failblog'],
        # ['github', 'https://github.com/guanguans/favorite-link/commits/master.atom'],
        # ['台灣最視覺系的全球要聞', 'https://dq.yam.com/rss.php'],
        # ['wa', 'https://wsgzao.github.io/atom.xml'],
        # ['开源中国社区', 'https://www.oschina.net/blog/rss'],
        ['CI0udG0d', 'http://feed.cnblogs.com/blog/u/550390/rss'],
        ['软件改变生活', 'https://feed.iplaysoft.com/'],
        ['背包客棧精選好文', 'https://www.backpackers.com.tw/forum/external.php'],
        ['不止是游戏', 'https://www.gcores.com/rss'],
        ['煎蛋', 'https://jandan.net/feed'],
        ['小众软件', 'https://www.appinn.com/feed/'],
        ['少数派', 'https://sspai.com/feed'],
        ['联合国新闻', 'https://news.un.org/feed/subscribe/zh/news/all/rss.xml'],
        ['36氪', 'https://36kr.com/feed'],
        ['cnBeta', 'https://rss.cnbeta.com/'],
        ['喷嚏网', 'http://www.dapenti.com/blog/rss2.asp'],
        ['糗事百科', 'https://www.qiushibaike.com/hot/rss']
    ]
    sql.clear_all('info')
    sql.clear_all('src')
    sql.src_addSrc(src_list)
    sql.info_reflashAll(src_name_list=None, all_force=True)
    # sql.info_getAll(uid)
    # sql.uts_addSrc()
    # sql.uts_removeSrc(uid, sid)
    # sql.test()


if __name__ == '__main__':
    main()
