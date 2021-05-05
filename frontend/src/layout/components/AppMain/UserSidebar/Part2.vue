<template>
  <div class="info-warpper">
    <el-card class="box-card" shadow="never">
      <el-row class="all-title">
        <el-col class="style2" :span="12">
          我的世界
        </el-col>
        <el-col class="addbutton" :span="12">
          <router-link to="/srcList">
            <el-button
                size="mini"
                type="primary"
                icon="el-icon-plus" circle>
            </el-button>
          </router-link>
        </el-col>
      </el-row>
      <el-row>
        <el-divider class="el-divider--horizontal"></el-divider>
      </el-row>
      <el-row>
        <el-table class="table-warpper" :data="tableData" :show-header="false">
          <el-table-column label="1" align="left">
            <template slot-scope="scope">
              <router-link to="/sb">{{ scope.row.src_name}}</router-link>
            </template>
          </el-table-column>
          <el-table-column label="2" align="right">
            <template slot-scope="scope">
              <el-button
                  size="mini"
                  type="danger"
                  icon="el-icon-minus"
                  @click="handleDelete(scope.$index, scope.row)" circle></el-button>
            </template>
          </el-table-column>
        </el-table>
      </el-row>
    </el-card>
  </div>
</template>

<script>
import { getSrc, deleteOne } from '@/api/userSidebar'
export default {
  name: "part2",
  data() {
    return {
      tableData: []
    }
  },
  created() {
    this.fetchData()
  },
  methods: {
    fetchData() {
      setTimeout(() => {
        getSrc({
          token: this.$store.getters.token
        }).then(response => {
          this.tableData = this.tableData.concat(response.body.content);
        });
      }, 10)
    },
    handleDelete(index, row) {
      setTimeout(() => {
        deleteOne({
          token: this.$store.getters.token,
          index:index,
          row:row,
          // eslint-disable-next-line no-unused-vars
        }).then(response => {
          this.tableData.splice(index, 1)
        });
      }, 10)
    },
    openDetails (row, event, column) {
      console.log(row, event, column)
      window.location.href="https://hao.360.com/?a1004"
    },
  }
}

</script>


<style scoped>
.info-warpper{
  list-style: none;
  display:flex;
  justify-content: center;
  margin-bottom: 18px;
  padding-left: 5px;
  font-size: 14px;
}
.box-card{
  width: 100%;
  border-radius: 20px;
}
.all-title{
  margin-left: 10px;
  margin-right: 10px;
}
.style2{
  text-align: left;
}
.addbutton{
  text-align: right;
}
.el-divider--horizontal{
  margin-top: 3px;
  margin-left: -20px;
  width: 120%;
}

.table-warpper{
  margin-top: -24px;

}
</style>
