// pages/baobiao/baobiao.js
var util = require('../../utils/util.js');
var app = getApp();
import * as echarts from '../../ec-canvas/echarts';
const db = wx.cloud.database()
function setOption(chart, ylist) {
  var options = {
    title: {
      left: 'center'
    },
    color: ["#37A2DA"],
    grid: {
      top: 20,
      right: 20,
      bottom: 30
    },
    tooltip: {
      show: true,
      trigger: 'axis'
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: ['1', '2', '3', '4', '5', '6', '7']
    },
    yAxis: {
      x: 'center',
      type: 'value',
      splitLine: {
        lineStyle: {
          type: 'dashed'
        }
      }
    },
    series: [{
      type: 'line',
      smooth: true,
      data: ylist
    }]
  }
  chart.setOption(options);
}

Page({

  /**
   * 页面的初始数据
   */
  data: {
    rawdata:[],
    //machineID:1,
    status:['正常','内环故障','外环故障'],    //jjjjjjjjjjjjjjjjjjjjjjjjj
    result:[],
    array: ['119', '186', '223'],
    array2: ['驱动端', '风扇端'],
    index:0,
    index2:0,
    i:0,
    time:'',
    timer:'',
    timer2: '',
    series: ["0.06636", "0.09595", "-0.11259", "0.01664", "0.09389","-0.01294", "0.13704"],
    //series: [],
    ylist: [],
    ec: {
      lazyLoad: true
    },
    macNum:3,                          //机组数量
    macData:['TEST1.csv','TEST11.csv','TEST21.csv'],         //模拟传感器数据
    //machineId:['100','108','121'],          //设备号
    machineId:['1','2','3'],          //测试设备号，对应集合
    location:['North-2-1','North_2-2','North-2-3'],  //设备位置
    latestDate:'2020/04/26',
    updated:''
  },
  onLoad: function () {
    this.setData({
      time: util.formatTime(new Date()),
    })
    this.setDatas(119)
    this.oneComponent = this.selectComponent('#mychart-dom-line');
    this.getOneOption(this.data.series);
  },
  //getDatas函数，调用api获取原始数据，参数（设备号，信息，回调结果）
  getDatas: function (macId, attr, callback) {
    wx.request({
      url: 'https://api.phmlearn.com/component/data/zhoucheng',
      method: 'POST',
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      data: {
        access_token: app.globalData.access_token,
        divice_id: macId,
        atrribute: attr
      },
      success: function (res) {
        callback(res)
      }
    })
  },
  //setArrData函数，设置“当前有效值”格式为小数点后5位
  setArrData: function (arr) {
    for (let i = 0; i < arr.length; i++) {
      arr[i] = arr[i].toFixed(5)
    }
    return arr
  },
  //setDatas函数，获取数据并整理结果为字典，参数（设备号）
  setDatas: function (macId) {
    this.getDatas(macId, 'DE_time', res => {
      this.setData({
        'result[0]': {
          key: '驱动端',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
    })
    this.getDatas(macId, 'FE_time', res => {
      this.setData({
        'result[1]': {
          key: '风扇端',
          max: Math.max(...res.data.data.data).toFixed(5),
          min: Math.min(...res.data.data.data).toFixed(5),
          arr: this.setArrData(res.data.data.data)
        }
      })
      this.startTimer()
      this.setDate()
    })

  },
  //startTimer函数，开始计时器
  startTimer: function () {
    //var that = this
    var index = this.data.index2
    var array = this.data.result[index].arr
    console.log(this.data.result[index].arr)
    console.log(index)
    this.setData({
      i: 0
    })
    this.setData({
      timer: setInterval(() => {
        if (this.data.i <= 3000) {
          this.setData({
            i: this.data.i + 1
          })   
          
          if(this.data.i%7 == 0){
            let series2 = []
            for (let n = this.data.i; n < this.data.i + 7; n++) {
              series2.push(array[n])
            }
            this.init_one(series2)
          }
          
        /*
          this.setData({
            series: series2
          })  */
          //console.log(this.data.series)
          //this.init_one(series2)

        }
        else {
          this.setData({
            i: 0
          })
          this.closeTimer(this.data.timer)
          this.closeTimer(this.data.timer2)
        }
      }, 1000)
    })
  },
  //setDate设置日期
  setDate: function () {
    this.setData({
      timer2: setInterval(() => {
        this.setData({
          time: util.formatTime(new Date())
        })
      }, 1000)
    })
  },
  closeTimer: function (time) {
    clearInterval(time)
  },
  init_one: function (ylist) {           //初始化第一个图表
    this.oneComponent.init((canvas, width, height) => {
      const chart = echarts.init(canvas, null, {
        width: width,
        height: height
      });
      setOption(chart, ylist)  //赋值给echart图表
      this.chart = chart;
      return chart;
    });
  },
  getChartdata: function (args) {
    let array = args
    let series1 = []
    for (let i = 0; i < 7; i++) {
      series1.push(array[i])
    }
    this.setData({
      series: series1
    })
  },
  getOneOption: function (series) {
    this.setData({
      ylist: series,
    })
    this.init_one(this.data.ylist)
  },
  bindPickerChange: function (e) {
    let arr=[119,186,223]
    this.closeTimer(this.data.timer)
    this.closeTimer(this.data.timer2)
    this.setData({
      index: e.detail.value 
    })
    let j = this.data.index
    //let fanid = this.data.fjnum[j]
    //this.getLabel(fanid)
    this.setDatas(arr[j])
    this.getOneOption(this.data.series);
  },
  bindPickerChange2: function (e) {
    this.setData({
      index2: e.detail.value
    })
    let index = e.detail.value
    let arr = this.data.result[index].arr
    this.getChartdata(arr)
    this.getOneOption(this.data.series)
  },
  navToInfo() {
    wx.navigateTo({
      url: '/pages/information/information',
    })
  },
  //获取传感器数据，实际生产时要更新这个接口，这里只是内置了3个数据集用于demo
  getSensorData:function(id){
    return this.data.macData[id]
  },
  //传感器数据特征提取
  feaSelect: function (filename,callback) {
    wx.request({
      url: 'https://api.phmlearn.com/component/upload/2/103',
      method: 'POST',
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      data: {
        access_token: app.globalData.access_token,
        file_name: filename,
      },
      success: function (res) {
        callback(res)
      }
    })
  },
  //诊断状态
  predict:function(filename,callback){
    wx.request({
      url: 'https://api.phmlearn.com/component/upload/ML/model/75/148',
      method: 'POST',
      header: {
        "Content-Type": "application/x-www-form-urlencoded"
      },
      data: {
        access_token: app.globalData.access_token,
        file_name: filename,
      },
      success: function (res) {
        callback(res)
      }
    })
  },
  //上传数据到云数据库
  uploadData:function(macNum,data){          //参数（macNum:设备序号0，1，2，data:当日数据）
    var that = this
    let id = this.data.machineId[macNum-1]  //设备号
    let loc = this.data.location[macNum]   //位置
    this.feaSelect(data , res =>{
      this.predict(res.data.data.file_name , res =>{
        db.collection(id).add({
          data: {
            HP:3,
            date:util.formatTime(new Date()),
            location: loc,
            status:res.data.data.predict[0],
            principal:app.globalData.principal,
          },
          success: function(res) {
            // res 是一个对象，其中有 _id 字段标记刚创建的记录的 id
            console.log(res)
            that.setData({
              updated:'今日已更新'
            })
          }
        })
      })
    })
  },
  //循环更新数据，每天由维护人员打卡记录
  dbRenew:function(e){
    let i = parseInt(this.data.machineId[this.data.index])      //每人运营三台机组
    let data = this.getSensorData(i)
    console.log(i)
    this.uploadData(i,data)      //数组下标0开始
    this.setData({
      latestDate:this.data.time
    })
    console.log(this.data.time)
  },
  onUnload: function () {
    clearInterval(this.data.timer2)
  }
})