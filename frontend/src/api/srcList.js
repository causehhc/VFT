import request from '@/utils/request'

export function getList(data) {
    return request({
        url: '/api/list/srcList',
        method: 'post',
        data
    })
}

export function addOne(data) {
    return request({
        url: '/api/userSidebar/addOne',
        method: 'post',
        data
    })
}

