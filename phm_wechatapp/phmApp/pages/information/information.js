// pages/information/information.js
const db = wx.cloud.database()

Page({

  /**
   * 页面的初始数据
   */
  data: {
    machineInfo:[],                   //1个机组的40行数据
    label: [],                           //故障类型
    informationdata:[]
  },

  /**
   * 生命周期函数--监听页面加载，此时调用query拿到40行数据
   */
  onLoad: function (options) {
    //保存this变量
    var _this = this;
    //调用云函数query
    wx.cloud.callFunction({
      // 云函数名称
      name: 'query',
      // 传给云函数的参数
      data: {
        //机组1，周一数据
        day: 1,
        machine: 1,
        label:0
      },
      //success函数为调用成功后的回调函数
      success: function (res) {
        console.log(res.result.data)
        _this.setData({
          informationdata:res.result.data,
          //label: res.result.data[0].label      //设置label值，显示在页面上。setData函数会触发页面的重新渲染
        })
      },
      //fail函数为调用失败后的回调函数
      fail: console.error
    })
  },

  /**
   * 生命周期函数--监听页面初次渲染完成
   */
  onReady: function () {

  },

  /**
   * 生命周期函数--监听页面显示
   */
  onShow: function () {
    /** 
    db.collection('machineData').where({
      day:1,
      machine:1
    })
    .get({
      success:function(res){
        console.log(res.data)
      }
    })
    */
  },

  /**
   * 生命周期函数--监听页面隐藏
   */
  onHide: function () {

  },

  /**
   * 生命周期函数--监听页面卸载
   */
  onUnload: function () {

  },

  /**
   * 页面相关事件处理函数--监听用户下拉动作
   */
  onPullDownRefresh: function () {

  },

  /**
   * 页面上拉触底事件的处理函数
   */
  onReachBottom: function () {

  },

  /**
   * 用户点击右上角分享
   */
  onShareAppMessage: function () {

  }
})