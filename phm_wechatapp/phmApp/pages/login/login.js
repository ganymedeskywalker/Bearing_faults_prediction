// phmApp/pages/login/login.js
const db = wx.cloud.database();
var app = getApp();
Page({

  /**
   * 页面的初始数据
   */
  data: {
    inputname:'',
    inputpassword:'',
    msg:''
  },
  getInputName: function (e) {
    this.setData({
      inputname:e.detail.value 
    })
  },
  getInputPW: function (e) {
    this.setData({
      inputpassword:e.detail.value  
    })
  },
  checkID: function(){
    var that = this      //函数第一层声明this，后面才能用this.setdata
    wx.cloud.callFunction({
      // 云函数名称
      name: 'query',
      // 传给云函数的参数
      data: {
        set:'staff',
        name:that.data.inputname,
        password:parseInt(that.data.inputpassword),
      },
      //success函数为调用成功后的回调函数
      success: function (res) {
        if(res.result.data.length!=0){
          app.globalData.macID = res.result.data[0].macID,
          app.globalData.position = res.result.data[0].position,
          app.globalData.principal = res.result.data[0].name,
          app.globalData.macSet = res.result.data[0].macSet,
          wx.switchTab({
            url: "../shikuang/shikuang"
          })
        }
        else{
          that.setData({
            msg:'信息错误'
          })
        }
      },
    })
  },
  /**
   * 生命周期函数--监听页面加载
   */
  onLoad: function (options) {

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