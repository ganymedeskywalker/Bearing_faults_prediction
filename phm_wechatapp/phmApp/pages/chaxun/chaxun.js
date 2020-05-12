// pages/chaxun/chaxun.js
const db = wx.cloud.database();//初始云
const _ = db.command

Page({
  data: {
    flag: true,
    inputday:'',
    inputmonth: '',
    inputyear: '',
  },
  getInput1: function (e) {
    this.setData({

      inputyear:e.detail.value //定义对象属性并把输入框数据赋值给它
     
    }),
    console.log(this.data.inputyear)
  },
  getInput2: function (e) {
    this.setData({

      inputmonth : e.detail.value //定义对象属性并把输入框数据赋值给它

    });
  },
  getInput3: function (e) {
    this.setData({

      inputday : e.detail.value //定义对象属性并把输入框数据赋值给它

    });
  },

  /**
   * 页面的初始数据
   */
  
  
  mysubmit: function () {
    //保存this变量
    var _this = this;
    //调用云函数query
    var a = parseInt(_this.data.inputyear);
    var b = parseInt(_this.data.inputmonth);
    var c = parseInt(_this.data.inputday);
    wx.cloud.callFunction({
      // 云函数名称
      name: 'query',
      // 传给云函数的参数
      data: {
        set:'machineData',
        year:a,//_this.data.inputyear,
        month: b,//_this.data.inputmonth,
        day:c//_this.data.inputday
      },
      //success函数为调用成功后的回调函数
      success: function (res) {
        console.log(res.result)
        console.log(_this.data.inputyear)
        console.log(_this.data.inputmonth)
        console.log(_this.data.inputday)
        _this.setData({
         chaxundata: res.result.data,
          flag: false
          //label: res.result.data[0].label      //设置label值，显示在页面上。setData函数会触发页面的重新渲染
        })
      },
      //fail函数为调用失败后的回调函数
      fail: console.error
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