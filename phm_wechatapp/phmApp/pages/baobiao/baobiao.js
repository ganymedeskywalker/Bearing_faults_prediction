// pages/baobiao/baobiao.js
var app = getApp();
const db = wx.cloud.database()
var util = require('../../utils/util.js');
Page({

  /**
   * 页面的初始数据
   */
  data: {
    staff:'czx',
    set:'100',
    result:[],
    status:['正常','球故障','内圈故障','外圈故障']
  },
  onLoad: function (options) {
    this.setData({
      //staff:app.globalData.principal,
      //set:app.globalData.macSet
    })
    this.queryData()
  },
  queryData:function(){
    var that = this
    db.collection(this.data.set).get({
      success:function(res){
        console.log(res.data)
        that.setData({
          result:res.data
        })
      }
    })
  },
  
 /*     */ 
})