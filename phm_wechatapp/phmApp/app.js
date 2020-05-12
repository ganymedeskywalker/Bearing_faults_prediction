//app.js
App({
  onLaunch: function () {
    
    //云开发初始化
    if (!wx.cloud) {
      console.error('请使用 2.2.3 或以上的基础库以使用云能力')
    } else {
      wx.cloud.init({
        //env: 'xyy666-9pxzd',
        env: 'test-yui',
      })
    }

    this.globalData = {
      access_token: "59f8142308974306b4466c952eab03fa.f810126d425daedcf77b158a87f38f83",
      macID:0,
      position:'',
      principal:'',
      macSet:'',
    }
  }
})
