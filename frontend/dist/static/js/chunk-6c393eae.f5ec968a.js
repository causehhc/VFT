(window["webpackJsonp"]=window["webpackJsonp"]||[]).push([["chunk-6c393eae"],{"05b1":function(t,s,i){"use strict";i.r(s);var e=function(){var t=this,s=t.$createElement,i=t._self._c||s;return i("div",{staticClass:"dynamic-container"},[i("ul",{directives:[{name:"infinite-scroll",rawName:"v-infinite-scroll",value:t.fetchData,expression:"fetchData"}],staticClass:"list",attrs:{"infinite-scroll-disabled":"disabled"}},t._l(t.list,(function(s,e){return i("li",{key:e,staticClass:"list-item"},[i("el-card",{staticClass:"box-card",attrs:{shadow:"hover"}},[i("div",{staticClass:"style1",attrs:{slot:"header"},slot:"header"},[i("span",[t._v(" "+t._s(s.postSrc)+" ")]),i("el-button",{staticStyle:{float:"right",padding:"3px 0"},attrs:{type:"text"}},[t._v(" 举报 ")])],1),i("div",{staticClass:"style2"},[i("span",[t._v(" "+t._s(s.postTitle)+" ")]),i("el-divider",{attrs:{direction:"vertical"}}),i("span",[t._v(" "+t._s(s.postUpdated)+" ")])],1),i("el-divider"),i("div",{staticClass:"style3"},[t._v(" "+t._s(s.postContent)+" ")]),i("div",{staticClass:"style4"},[i("el-button",{staticStyle:{float:"right",padding:"3px 0"},attrs:{type:"text"},on:{click:function(i){return t.addLikes(s,s.postID)}}},[t._v(" 一键爱国 ")]),"0"!==s.postLikes?i("div",{staticStyle:{float:"right",padding:"3px 0"},attrs:{type:"text"}},[t._v(" "+t._s(s.postLikes)+" ")]):t._e()],1)],1)],1)})),0),i("div",{staticClass:"OtherNote"},[t.listLoading?i("p",{staticClass:"listLoading",staticStyle:{"margin-top":"10px"}},[i("span")]):t._e(),t.noMore?i("p",{staticStyle:{"margin-top":"10px","font-size":"13px",color:"#ccc"}},[t._v("没有更多了")]):t._e()])])},a=[],n=(i("99af"),i("b775"));function o(t){return Object(n["a"])({url:"/api/list/dynamic",method:"post",data:t})}function c(t){return Object(n["a"])({url:"/api/list/likes",method:"post",data:t})}var l={name:"index",data:function(){return{count:0,listLoading:!1,totalPages:"",list:[]}},computed:{noMore:function(){return this.count>=this.totalPages-1},disabled:function(){return this.listLoading||this.noMore}},created:function(){this.fetchData()},methods:{fetchData:function(){var t=this;this.listLoading=!0,setTimeout((function(){t.count+=1,o({count:t.count,token:t.$store.getters.token}).then((function(s){t.list=t.list.concat(s.body.content),t.totalPages=s.body.totalPages,t.listLoading=!1}))}),10)},addLikes:function(t,s){t.postLikes=parseInt(t.postLikes)+1,setTimeout((function(){c({ID:s}).then((function(t){console.log(t.body.data)}))}),10)}}},r=l,d=(i("37f6"),i("2877")),u=Object(d["a"])(r,e,a,!1,null,"686e330e",null);s["default"]=u.exports},"37f6":function(t,s,i){"use strict";i("78d0")},"78d0":function(t,s,i){}}]);