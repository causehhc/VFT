import request from '@/utils/request'

export function getSrc(data) {
    return request({
        url: '/api/userSidebar/getSrc',
        method: 'post',
        data
    })
}

export function deleteOne(data) {
    return request({
        url: '/api/userSidebar/deleteOne',
        method: 'post',
        data
    })
}
