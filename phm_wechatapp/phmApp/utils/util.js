const formatTime = date => {
  const year = date.getFullYear()
  const month = date.getMonth() + 1
  const day = date.getDate()
  const hour = date.getHours()
  const minute = date.getMinutes()
  const second = date.getSeconds()

  //return [year, month, day].map(formatNumber).join('/') + ' ' + [hour, minute, second].map(formatNumber).join(':')
  return [year, month, day].map(formatNumber).join('/')
}

const formatNumber = n => {
  n = n.toString()
  return n[1] ? n : '0' + n
}

function reqFunc(url,data,callback){       //nextPage没用
  wx.showLoading({
    title: '正在处理中',
  })
  wx.request({
    url: url,
    method: "POST",
    header: {
      "Content-Type": "application/x-www-form-urlencoded"
    },
    data:data,
    complete(res){
      wx.hideLoading();
      if(res.data.status == 0){
        wx.showToast({
          title: '处理成功'
        })
      }else{
        wx.showToast({
          title: '处理失败',
          icon:"none"
        })
      };
      /*
      setTimeout(function () {
        wx.navigateTo({
          url: '/pages/' + nextPage + '/' + nextPage,
        })
      }, 2000)
      */
    },
    success(res){
      callback(res);
    }
    })
}

module.exports = {
  formatTime: formatTime,
  reqFunc: reqFunc
}
