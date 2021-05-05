import request from '@/utils/request'

export function getList(data) {
    return request({
        url: '/api/list/dynamic',
        method: 'post',
        data
    })
}

export function postLikes(data) {
    return request({
        url: '/api/list/likes',
        method: 'post',
        data
    })
}
