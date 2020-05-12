// pages/information/information.js
var app = getApp();
const db = wx.cloud.database()
var util = require('../../utils/util.js');
Page({

  /**
   * 页面的初始数据
   */
  data: {
    machineInfo: ['100','108','121','133','147','160','172','188','200'],                   //1个机组的40行数据
    label: [],                           //故障类型
    date: '',
    status:[0,0,0,0,0,0,0,0,0],
    faults:[],
    fnum:0,        //故障数
    allnum:9,         //总机组数
  },

  /**
   * 生命周期函数--监听页面加载，此时调用query拿到40行数据
   */
  onLoad: function (options) {
    this.setData({
      date:util.formatTime(new Date()),
      //date:'2020/04/26'  //暂时设置
    })
    this.getAllStatus()
    //this.getFault()
  },
  //获取一个集合当天的故障类型
  getMacStatus:function(info){
    var that = this
    var set = that.data.machineInfo[info]
    wx.cloud.callFunction({
      name: 'queryMac',
      data: {
        set:set,
        date:that.data.date,
      },
      success: function (res) {
        //let series = that.data.status 
        //series[info] = res.result.data[0].predict 
        //that.setData({
        //  'status':series
        //})
        let s = res.result.data[0].predict 
        let num = that.data.fnum
        let temp = 'faults[' + info +']'
        if(s == 1){
          that.setData({
            [temp]:{
              faultn:1,
              fault:'球故障',
              principal:res.result.data[0].principal,
              loc:res.result.data[0].location,
              hp:3
            },
            fnum:num + 1
          })
        }
        else if(s == 2){
          that.setData({
            [temp]:{
              faultn:2,
              fault:'内圈故障',
              principal:res.result.data[0].principal,
              loc:res.result.data[0].location,
              hp:3
            },
            fnum:num + 1
          })
        }
        else if(s == 3){
          that.setData({
            [temp]:{
              faultn:3,
              fault:'外圈故障',
              principal:res.result.data[0].principal,
              loc:res.result.data[0].location,
              hp:3
            },
            fnum:num + 1
          })
        }
        else{
          that.setData({
            [temp]:{
              faultn:0,
              fault:'正常',
              principal:res.result.data[0].principal,
              loc:res.result.data[0].location,
              hp:3
            }
          })
        }
      }
    })
  },
  //循环获取9个集合的状态
  getAllStatus:function(){
    //var that = this
    var macSet = this.data.machineInfo
    for(let i = 0;i < macSet.length;i++){
      this.getMacStatus(i)
    }
  },
  getFault:function(){
    //var that = this
    var macSet = this.data.machineInfo
    var s = this.data.status
    for(let i = 0;i < macSet.length;i++){
      let index = 0
      console.log(s)
      
    }
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

/*
   //保存this变量
    var _this = this;
    //调用云函数query
    wx.cloud.callFunction({
      // 云函数名称
      name: 'baojing',
      // 传给云函数的参数
      data: {
        //机组1，周一数据
        
        label:0
      },
      //success函数为调用成功后的回调函数
      success: function (res) {
        //console.log(res.result)
        _this.setData({
          baojingdata: res.result.data,
          //label: res.result.data[0].label      //设置label值，显示在页面上。setData函数会触发页面的重新渲染
        })
      },
      //fail函数为调用失败后的回调函数
      fail: console.error
    }),
      wx.cloud.callFunction({
        // 云函数名称
        name: 'baojing',
        // 传给云函数的参数
        data: {
          //机组1，周一数据

          label: 2
        },
        //success函数为调用成功后的回调函数
        success: function (res) {
          //console.log(res.result)
          _this.setData({
            baojingdata2: res.result.data,
            //label: res.result.data[0].label      //设置label值，显示在页面上。setData函数会触发页面的重新渲染
          })
        },
        //fail函数为调用失败后的回调函数
        fail: console.error
      }),
      wx.cloud.callFunction({
        // 云函数名称
        name: 'baojing',
        // 传给云函数的参数
        data: {
          //机组1，周一数据

          label: 3
        },
        //success函数为调用成功后的回调函数
        success: function (res) {
          //console.log(res.result)
          _this.setData({
            baojingdata3: res.result.data,
            //label: res.result.data[0].label      //设置label值，显示在页面上。setData函数会触发页面的重新渲染
          })
        },
        //fail函数为调用失败后的回调函数
        fail: console.error
      })
      */