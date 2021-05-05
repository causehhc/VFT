import request from '@/utils/request'

export function getList(data) {
    return request({
        url: '/api/list/export',
        method: 'post',
        data
    })
}
