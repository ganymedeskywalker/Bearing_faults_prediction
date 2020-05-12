// pages/day/day.js

const db = wx.cloud.database();//初始云
const _ = db.command

//time获取时间
var timestamp = Date.parse(new Date()); //返回当前时间毫秒数
timestamp = timestamp / 1000;
//获取当前时间
var n = timestamp * 1000;
var date = new Date(n);
//年  
var Y = date.getFullYear();
//月 
var M = (date.getMonth() + 1 < 10 ? '0' + (date.getMonth() + 1) : date.getMonth() + 1);
//日
var D = date.getDate() < 10 ? '0' + date.getDate() : date.getDate();
//时
var h = date.getHours();
//分
var m = date.getMinutes();
//秒
var s = date.getSeconds();

Page({

  /**
   * 页面的初始数据
   */
  data: {

  },

  /**
   * 生命周期函数--监听页面加载
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

       day:D
      },
      //success函数为调用成功后的回调函数
      success: function (res) {
        console.log(res.result)
        _this.setData({
          baobiaodata: res.result.data,
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