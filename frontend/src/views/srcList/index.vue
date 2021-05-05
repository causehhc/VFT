<template>
  <div class="anls-container">
    <ul class="list" v-infinite-scroll="fetchData" infinite-scroll-disabled="disabled">
      <li v-for="(i,index) in list" class="list-item" :key="index">
        <el-card class="box-card" shadow="hover">
          <div slot="header" class="style1">
            <span>
              {{i.sname}}
            </span>
            <el-button
                size="mini"
                type="primary"
                icon="el-icon-plus"
                @click="handleAdd(i.sid)" circle>
            </el-button>
          </div>
<!--          <div class="text-item">-->
<!--            {{i.ssummer}}-->
<!--          </div>-->
<!--          <div class="img-item">-->
<!--            <img src="https://shadow.elemecdn.com/app/element/hamburger.9cf7b091-55e9-11e9-a976-7f4d0b07eef6.png" class="image">-->
<!--          </div>-->
        </el-card>
      </li>
    </ul>
    <div class="OtherNote">
      <p v-if="listLoading" style="margin-top:10px;" class="listLoading">
        <span></span>
      </p>
      <p v-if="noMore" style="margin-top:10px;font-size:13px;color:#ccc">没有更多了</p>
    </div>
  </div>
</template>

<script>
import { getList, addOne } from '@/api/srcList'
export default {
  name: "index",
  data() {
    return {
      count: 0,//起始页数值为0
      listLoading: false,
      totalPages: "",//取后端返回内容的总页数
      list: [] //后端返回的数组
    };
  },
  computed: {
    noMore() {
      //当起始页数大于总页数时停止加载
      return this.count >= this.totalPages - 1;
    },
    disabled() {
      return this.listLoading || this.noMore;
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    fetchData() {
      this.listLoading = true
      setTimeout(() => {
        this.count += 1;
        getList({
          count: this.count,
        }).then(response => {
          this.list = this.list.concat(response.body.content); //因为每次后端返回的都是数组，所以这边把数组拼接到一起
          this.totalPages = response.body.totalPages;
          this.listLoading = false
        });
      }, 10)
    },
    handleAdd(sid) {
      console.log(1);
      setTimeout(()=>{
        addOne({
          token: this.$store.getters.token,
          sid:sid
        }).then(response => {
          console.log(response)
        });
      }, 10)
    },
  }
}
</script>

<style scoped>
.list {
  margin: 0 auto;
  padding-right: 5px;
  font-size: 14px;
}
.list-item {
  list-style: none;
  display:flex;
  justify-content: center;
  margin-bottom: 18px;
}
.box-card {
  width: 100%;
  border-radius: 20px
}

.image {
  width: 33%;
  margin-top: 5px;
  margin-right: 5px;
  border-radius: 10px
}

.OtherNote{
  text-align: center;
}
.loading span {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid #409eff;
  border-left: transparent;
  animation: zhuan 0.5s linear infinite;
  border-radius: 50%;
}
@keyframes zhuan {
  0% {
    transform: rotate(0);
  }
  100% {
    transform: rotate(360deg);
  }
}

.style1:before,
.style1:after {
  display: table;
  content: "";
}
.style1:after {
  clear: both
}
</style>
