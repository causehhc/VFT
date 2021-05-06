import time
from datetime import datetime


def transTimeFormat(str):
    tt = str.split()
    if len(tt) > 1:
        tt.pop(-1)
    tt = ' '.join(tt)

    t = tt
    tex = [
        '%a, %d %b %Y %H:%M:%S',  # Sun, 05 May 2019 00:00:00
        '%Y-%m-%d %H:%M:%S',  # 2021-04-16 16:40:32
        '%Y-%m-%dT%H:%M:%SZ',  # 2021-03-26T04:43:00Z
    ]
    d = None
    for j in range(len(tex)):
        try:
            d = datetime.strptime(t.strip(), tex[j])
            # d = datetime.strptime(time.strip(),tex[3])
            break
        except ValueError:
            continue

    stamp = datetime.timestamp(d)
    time_local = time.localtime(stamp)
    dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
    return dt


if __name__ == '__main__':

    # gmt = "Sun, 05 May 2019 00:00:00 GMT"
    # gmt = "Wed, 05 May 2021 05:17:42 +0000"
    # gmt = "2021-04-16 16:40:32  -0000"
    # gmt = "2021-03-26T04:43:00Z"
    print(transTimeFormat(gmt))
