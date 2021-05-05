<template>
  <div class="info-warpper">
    <el-card class="box-card" shadow="never">
      <el-row>
        <el-col :span="24">
          <el-avatar src="https://cube.elemecdn.com/0/88/03b0d39583f48206768a7534e55bcpng.png"></el-avatar>
        </el-col>
      </el-row>
      <el-row>
        <el-col :span="24">
          username
        </el-col>
      </el-row>
      <el-row>
        <el-col :span="12">
          0 followers
        </el-col>
        <el-col :span="12">
          0 following
        </el-col>
      </el-row>
      <el-row>
        <el-col :span="24">
          0 src
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<script>
export default {
  methods: {sb(){
      console.log(1)
      window.location.href="https://hao.360.com/?a1004"
    }},
  data() {
    return {
      fits: [''],
      url: 'https://fuss10.elemecdn.com/e/5d/4a731a90594a4af544c0c25941171jpeg.jpeg'
    }
  },

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


</style>
